# Compute Distance / Divergence between Distributions
import numpy as np
import pandas as pd
import os, glob
import sqlite3 as sql
import itertools as it

from numpy.random import gamma, normal
from scipy.linalg import cholesky
from random import sample
from collections import namedtuple

from multiprocessing import Pool

from argparser import argparser_ppl as argparser
from hypercube_deviance import energy_score_euclidean, energy_score_hypercube
from energy import energy_score
import cUtility as cu
import models, models_mpi
import model_dppn as dppn

Results = {**models.Results, **models_mpi.Results}

from data import Data, euclidean_to_hypercube, euclidean_to_simplex, angular_to_euclidean

np.seterr(under = 'ignore')

epsilon = 1e-30

# Object defining loss criterion
class PostPredLoss(object):
    def L1(self, data):
        data2 = np.empty(data.shape)
        for n in range(data.shape[0]):
            data2[n] = (data[n].T / data[n].sum(axis = 1)).T
        return data2

    def L2(self, data):
        data2 = np.empty(data.shape)
        for n in range(data.shape[0]):
            data2[n] = (data[n].T / np.sqrt((data[n] * data[n]).sum(axis = 1))).T
        return data2

    def Linf(self, data):
        data2 = np.empty(data.shape)
        for n in range(data.shape[0]):
            data2[n] = (data[n].T / data[n].max(axis = 1)).T
        return data2

    def __postpredloss(self, predicted, empirical):
        pmean = predicted.mean(axis = 0)
        pdiff = pmean - empirical
        avg_bias = ((pdiff * pdiff).sum(axis = 1)).mean()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        avg_vari = pvari.mean()
        return avg_bias + avg_vari

    def posterior_predictive_loss_L1(self):
        predicted = self.L1(self.prediction())
        return self.__postpredloss(predicted, euclidean_to_simplex(self.data.Yl))

    def posterior_predictive_loss_L2(self):
        predicted = self.L2(self.prediction())
        return self.__postpredloss(predicted, self.data.Yl)

    def posterior_predictive_loss_Linf(self):
        predicted = self.Linf(self.prediction())
        return self.__postpredloss(predicted, euclidean_to_hypercube(self.data.Yl))

    def energy_score_L1(self):
        predicted = self.L1(self.prediction())
        return energy_score_euclidean(predicted, euclidean_to_simplex(self.data.Yl))

    def energy_score_L2(self):
        predicted = self.L2(self.prediction())
        res = energy_score_euclidean(
                np.moveaxis(predicted, 0, 1),
                self.data.Yl
                )
        return res

    def energy_score_Linf(self):
        predicted = self.Linf(self.prediction())
        return energy_score(np.moveaxis(predicted, 0, 1), euclidean_to_hypercube(self.data.Yl))

# Object defining how predictive distribution is assembled.
# All of the gamma-based models share a basic prediction method
class Prediction_Gamma_Vanilla_Restricted(object):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            zeta = self.samples.zeta[s]
            predicted[s] = gamma(shape = zeta, size = (self.nDat, self.nCol)) + epsilon
        return predicted

class Prediction_Gamma_Vanilla(object):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            zeta = self.samples.zeta[s]
            sigma = self.samples.sigma[s]
            predicted[s] = gamma(shape = zeta, scale = 1/sigma, size = (self.nDat, self.nCol)) + epsilon
        return predicted

class Prediction_Gamma_Restricted(object):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            zeta = self.samples.zeta[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = zeta) + epsilon
        return predicted

class Prediction_Gamma(object):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            zeta = self.samples.zeta[s][self.samples.delta[s]]
            sigma = self.samples.sigma[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = zeta, scale = 1/sigma) + epsilon
        return predicted

class Prediction_Gamma_Alter(object):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta = self.samples.beta[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = alpha, scale = 1/beta) + epsilon
        return predicted

class Prediction_Gamma_Alter_Restricted(object):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = alpha) + epsilon
        return predicted

# Special case for Probit Normal model
class DPPN_Result(dppn.Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol - 1))
        for i in range(self.nSamp):
            pred_temp = np.empty((self.nDat, self.nCol - 1))
            for j in range(self.nDat):
                pred_temp[j] = 0.5 * np.pi * self.invprobit(
                    + self.samples.mu[i][self.samples.delta[i,j]]
                    + cholesky(self.samples.Sigma[i][self.samples.delta[i,j]]) @ normal(size = self.nCol - 1)
                    )
            predicted[i] = angular_to_euclidean(pred_temp)
        return predicted

Results['dppn'] = DPPN_Result

# Updating the Result objects with the new methods.

Prediction_Gammas = {}
for model in ['md','dpd','mprg','dpprg']:
    Prediction_Gammas[model] = Prediction_Gamma_Restricted
for model in ['mgd','dpgd','mpg','dppg']:
    Prediction_Gammas[model] = Prediction_Gamma
for model in ['vd','vprg']:
    Prediction_Gammas[model] = Prediction_Gamma_Vanilla_Restricted
for model in ['vgd','vpg']:
    Prediction_Gammas[model] = Prediction_Gamma_Vanilla
for model in ['mdln','dpdln','mprgln', 'dpprgln']:
    Prediction_Gammas[model] = Prediction_Gamma_Alter_Restricted
for model in ['mgdln','dpgdln','mpgln','dppgln']:
    Prediction_Gammas[model] = Prediction_Gamma_Alter
for model in ['dppn']:
    Prediction_Gammas[model] = object

def ResultFactory(model, path):
    class Result(Results[model], PostPredLoss, Prediction_Gammas[model]):
        pass
    return Result(path)

PPLResult = namedtuple('PPLResult', 'type name PPL_L1 PPL_L2 PPL_Linf ES_Linf')

def ppl_generation(model):
    result = ResultFactory(*model)
    pplr = PPLResult(
        model[0],
        os.path.splitext(os.path.split(model[1])[1])[0],
        result.posterior_predictive_loss_L1(),
        result.posterior_predictive_loss_L2(),
        result.posterior_predictive_loss_Linf(),
        result.energy_score_Linf(),
        )
    return pplr

if __name__ == '__main__':
    args = argparser()
    model_types = sorted(Prediction_Gammas.keys())

    models = []
    for model_type in model_types:
        mm = glob.glob(os.path.join(args.path, model_type, 'results*.db'))
        for m in mm:
            models.append((model_type, m))

    pplrs = []
    for model in models:
        print('Processing model {}   '.format(model[0]), end = ' ')
        try:
            pplrs.append(ppl_generation(model))
            print('Passed')
        except pd.io.sql.DatabaseError:
            print('Failed')
            pass

    df = pd.DataFrame(
        pplrs,
        columns = ('type','name','PPL_L1','PPL_L2','PPL_Linf','ES_Linf'),
        )
    df.to_csv(os.path.join(args.path, 'post_pred_loss_results.csv'), index = False)

# EOF
