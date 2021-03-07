# Compute Distance / Divergence between Distributions
import numpy as np
import pandas as pd
import os, glob
import sqlite3 as sql
import itertools as it
from scipy.stats import gamma, norm as normal
from scipy.linalg import cholesky
from random import sample
from collections import namedtuple

import simplex as smp
import m_projgamma as mpg
import dp_projgamma as dpmpg
import dp_pgln as dppgln
from distance import energy_score

import cUtility as cu

from data import Data, euclidean_to_hypercube, euclidean_to_simplex

np.seterr(under = 'ignore')

epsilon = 1e-20

class FMIX_Result(smp.FMIX_Result):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            eta = self.samples.eta[s][self.samples.delta[s]]
            predicted[s] = gamma.rvs(a = eta) + epsilon
        return predicted

    def L1(self, data):
        data2 = np.empty(data.shape)
        for n in data.shape[0]:
            data2[n] = (data[n].T / data[n].sum(axis = 1)).T
        return data2

    def L2(self, data):
        data2 = np.empty(data.shape)
        for n in data.shape[0]:
            data2[n] = (data[n].T / np.sqrt((data[n] * data[n]).sum(axis = 1))).T
        return data2

    def Linf(self, data):
        data2 = np.empty(data.shape)
        for n in data.shape[0]:
            data2[n] = (data[n].T / data[n].max(axis = 1)).T
        return data2

    def posterior_predictive_loss_L1(self):
        # predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        # for s in range(self.nSamp):
        #     eta = self.samples.eta[s][self.samples.delta[s]]
        #     temp = gamma.rvs(a = eta) + 1e-20
        #     predicted[s] = (temp.T / temp.sum(axis = 1)).T
        predicted = self.L1(self.prediction())
        pmean = predicted.mean(axis = 0)
        pdiff = pmean - euclidean_to_simplex(self.data.Yl)
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def posterior_predictive_loss_L2(self):
        # predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        # for s in range(self.nSamp):
        #     eta = self.samples.eta[s][self.samples.delta[s]]
        #     temp = gamma.rvs(a = eta) + 1e-20
        #     predicted[s] = (temp.T / np.sqrt((temp * temp).sum(axis = 1))).T
        predicted = self.L2(self.prediction())
        pmean = predicted.mean(axis = 0)
        pdiff = pmean - self.data.Yl
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def posterior_predictive_loss_Linf(self):
        # predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        # for s in range(self.nSamp):
        #     eta = self.samples.eta[s][self.samples.delta[s]]
        #     temp = gamma.rvs(a = eta) + 1e-20
        #     predicted[s] = (temp.T / temp.max(axis = 1)).T
        predicted = self.Linf(self.prediction())
        pmean = predicted.mean(axis = 0)
        pdiff = pmean - euclidean_to_hypercube(self.data.Yl)
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def energy_score(self, predicted, empirical):
        return energy_score(predicted, empirical)

    def energy_score_L1(self):
        predicted = self.L1(self.prediction())
        return energy_score(predicted, euclidean_to_simplex(self.data.Yl))

    def energy_score_L2(self):
        predicted = self.L2(self.prediction())
        return energy_score(predicted, self.data.Yl)

    def energy_score_Linf(self):
        predicted = self.Linf(self.prediction())
        return energy_score(predicted, self.data.Yl)

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MPG_Result(mpg.MPGResult):
    def posterior_predictive_loss_L1(self):
        """ Computes Posterior Predictive Loss (Gelfand and Ghosh) on the posterior predictive
        distribution generated on the L1 Simplex (still in euclidean coords) """
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            temp = gamma.rvs(a = alpha, scale = 1/beta) + 1e-20
            predicted[s] = (temp.T / temp.sum(axis = 1)).T

        pmean = predicted.mean(axis = 0)
        pdiff = pmean - euclidean_to_simplex(self.data.Yl)
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def posterior_predictive_loss_L2(self):
        """ Computes Posterior Predictive Loss (Gelfand and Ghosh) on the posterior predictive
        distribution generated on the L2 hypersphere (still in euclidean coords) """
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            temp = gamma.rvs(a = alpha, scale = 1/beta) + 1e-20
            predicted[s] = (temp.T / np.sqrt((temp * temp).sum(axis = 1))).T

        pmean = predicted.mean(axis = 0)
        pdiff = pmean - self.data.Yl
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def posterior_predictive_loss_Linf(self):
        """ Computes Posterior Predictive Loss (Gelfand and Ghosh) on the posterior predictive
        distribution generated on the L2 hypersphere (still in euclidean coords) """
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            temp = gamma.rvs(a = alpha, scale = 1/beta) + 1e-20
            predicted[s] = (temp.T / temp.max(axis = 1)).T

        pmean = predicted.mean(axis = 0)
        pdiff = pmean - euclidean_to_hypercube(self.data.Yl)
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class DPMPG_Result(dpmpg.ResultDPMPG):
    def posterior_predictive_loss_L1(self):
        """ Computes Posterior Predictive Loss (Gelfand and Ghosh) on the posterior predictive
        distribution generated on the L1 Simplex (still in euclidean coords) """
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            temp = gamma.rvs(a = alpha, scale = 1/beta) + 1e-20
            predicted[s] = (temp.T / temp.sum(axis = 1)).T

        pmean = predicted.mean(axis = 0)
        pdiff = pmean - euclidean_to_simplex(self.data.Yl)
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def posterior_predictive_loss_L2(self):
        """ Computes Posterior Predictive Loss (Gelfand and Ghosh) on the posterior predictive
        distribution generated on the L2 hypersphere (still in euclidean coords) """
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            temp = gamma.rvs(a = alpha, scale = 1/beta) + 1e-20
            predicted[s] = (temp.T / np.sqrt((temp * temp).sum(axis = 1))).T

        pmean = predicted.mean(axis = 0)
        pdiff = pmean - self.data.Yl
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def posterior_predictive_loss_Linf(self):
        """ Computes Posterior Predictive Loss (Gelfand and Ghosh) on the posterior predictive
        distribution generated on the L2 hypersphere (still in euclidean coords) """
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            temp = gamma.rvs(a = alpha, scale = 1/beta) + 1e-20
            predicted[s] = (temp.T / temp.max(axis = 1)).T

        pmean = predicted.mean(axis = 0)
        pdiff = pmean - euclidean_to_hypercube(self.data.Yl)
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class DPPGLN_Result(dppgln.DPMPG_Result):
    def posterior_predictive_loss_L1(self):
        """ Computes Posterior Predictive Loss (Gelfand and Ghosh) on the posterior predictive
        distribution generated on the L1 Simplex (still in euclidean coords) """
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            temp = gamma.rvs(a = alpha, scale = 1/beta) + 1e-20
            predicted[s] = (temp.T / temp.sum(axis = 1)).T

        pmean = predicted.mean(axis = 0)
        pdiff = pmean - euclidean_to_simplex(self.data.Yl)
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def posterior_predictive_loss_L2(self):
        """ Computes Posterior Predictive Loss (Gelfand and Ghosh) on the posterior predictive
        distribution generated on the L2 hypersphere (still in euclidean coords) """
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            temp = gamma.rvs(a = alpha, scale = 1/beta) + 1e-20
            predicted[s] = (temp.T / np.sqrt((temp * temp).sum(axis = 1))).T

        pmean = predicted.mean(axis = 0)
        pdiff = pmean - self.data.Yl
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def posterior_predictive_loss_Linf(self):
        """ Computes Posterior Predictive Loss (Gelfand and Ghosh) on the posterior predictive
        distribution generated on the L2 hypersphere (still in euclidean coords) """
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            temp = gamma.rvs(a = alpha, scale = 1/beta) + 1e-20
            predicted[s] = (temp.T / temp.max(axis = 1)).T

        pmean = predicted.mean(axis = 0)
        pdiff = pmean - euclidean_to_hypercube(self.data.Yl)
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            # pvari[d] = np.linalg.det(np.cov(pdevi[:,d].T))
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()

        return (pplos, ppvar)

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

Result = {
    'fmix'   : FMIX_Result,
    'mpg'    : MPG_Result,
    'dpmpg'  : DPMPG_Result,
    'dppgln' : DPPGLN_Result,
    }

if __name__ == '__main__':
    base_path = './output'
    model_types = ['fmix','dpmpg','dppgln','mpg'] #,'dpmp']

    models = []
    for model_type in model_types:
        mm = glob.glob(os.path.join(base_path, model_type, 'results_*.db'))
        for m in mm:
            models.append((model_type, m))

    PostPredLossResult = namedtuple('PostPredLossResult', 'type name L1_G L1_P L2_G L2_P Linf_G Linf_P')
    postpredlossresults = []
    for model in models:
        result = Result[model[0]](model[1])
        postpredlossresults.append(
            PostPredLossResult(
                model[0],
                os.path.splitext(os.path.split(model[1])[1])[0],
                *result.posterior_predictive_loss_L1(),
                *result.posterior_predictive_loss_L2(),
                *result.posterior_predictive_loss_Linf(),
                )
            )
        del(result)

    df = pd.DataFrame(
        postpredlossresults,
        columns = ('type','name','L1_G','L1_P','L2_G','L2_P','Linf_G','Linf_P'),
        )
    df.to_csv('./output/post_pred_loss_results.csv', index = False)

# EOF
