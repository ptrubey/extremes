# Compute Distance / Divergence between Distributions
import numpy as np
import pandas as pd
import os, glob
import sqlite3 as sql
import itertools as it
#from scipy.stats import gamma, norm as normal
from numpy.random import gamma, normal
from scipy.linalg import cholesky
from random import sample
from collections import namedtuple

# import simplex as smp
# import m_projgamma as mpg
# import dp_projgamma as dpmpg
# import dp_rprojgamma as dpmrpg
# import dp_pgln as dppgln
import model_dirichlet as d
import model_gendirichlet as gd
import model_projgamma as pg
import model_projresgamma as prg
import model_dln as dln
import model_gdln as gdln
import model_pgln as pgln
import model_prgln as prgln
import model_probit as p

from hypercube_deviance import energy_score_euclidean, energy_score_hypercube
from energy import energy_score
# from distance import energy_score

import cUtility as cu

from data import Data, euclidean_to_hypercube, euclidean_to_simplex

np.seterr(under = 'ignore')

epsilon = 1e-30

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
        pplos = ((pdiff * pdiff).sum(axis = 1)).sum()

        pdevi = predicted - pmean
        pvari = np.empty(self.nDat)
        for d in range(self.nDat):
            pvari[d] = np.trace(np.cov(pdevi[:,d].T))
        ppvar = pvari.sum()
        return pplos + ppvar

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

class DPD_Result(d.DPD_Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            zeta = self.samples.zeta[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = zeta) + epsilon
        return predicted

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MD_Result(d.MDResult, DPD_Result):
    pass

class DPPRG_Result(prg.DPPRG_Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            zeta = self.samples.zeta[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = zeta) + epsilon
        return predicted

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MPRG_Result(prg.MPRG_Result, DPPRG_Result):
    pass

class DPGD_Result(gd.DPGD_Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            zeta = self.samples.zeta[s][self.samples.delta[s]]
            sigma = self.samples.sigma[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = zeta, scale = 1 / sigma) + epsilon
        return predicted

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MGD_Result(gd.MGD_Result, DPGD_Result):
    pass

class DPPG_Result(pg.DPPG_Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            zeta = self.samples.zeta[s][self.samples.delta[s]]
            sigma = self.samples.sigma[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = zeta, scale = 1 / sigma) + epsilon
        return predicted

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MPG_Result(pg.MPG_Result, DPPG_Result):
    pass

class DPPGLN_Result(pgln.DPPGLN_Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.beta[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = alpha, scale = 1 / beta) + epsilon
        return predicted

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MPGLN_Result(pgln.MPGLN_Result, DPPGLN_Result):
    pass

class DPPRGLN_Result(prgln.DPPRGLN_Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = alpha) + epsilon
        return predicted

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MPRGLN_Result(prgln.MPRGLN_Result, DPPRGLN_Result):
    pass

class DPGDLN_Result(gdln.DPGDLN_Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            beta  = self.samples.alpha[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = alpha, scale = 1/beta) + epsilon
        return predicted

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MGDLN_Result(gdln.MGDLN_Result, DPGDLN_Result):
    pass

class DPDLN_Result(dln.DPDLN_Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for s in range(self.nSamp):
            alpha = self.samples.alpha[s][self.samples.delta[s]]
            predicted[s] = gamma(shape = alpha) + epsilon
        return predicted

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MDLN_Result(dln.MDLN_Result, DPDLN_Result):
    pass

class DPPN_Result(p.DPPN_Result, PostPredLoss):
    def prediction(self):
        predicted = np.empty((self.nSamp, self.nDat, self.nCol))
        for i in range(self.nSamp):
            for j in range(self.nDat):
                predicted[i,j] = 0.5 * np.pi * self.invprobit(
                    + self.samples.mu[i][self.samples.delta[i,j]]
                    + cholesky(self.samples.Sigma[i][self.samples.delta[i,j]]) @ normal(size = self.nCol)
                    )
        return angular_to_euclidean(predicted)

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

# class MPN_Result(DPPN_Result):
#     pass

# class FMIX_Result(smp.FMIX_Result, PostPredLoss):
#     def prediction(self):
#         predicted = np.empty((self.nSamp, self.nDat, self.nCol))
#         for s in range(self.nSamp):
#             eta = self.samples.eta[s][self.samples.delta[s]]
#             predicted[s] = gamma.rvs(a = eta) + epsilon
#         return predicted
#
#     def __init__(self, path):
#         super().__init__(path)
#         self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
#         return
#
# class DPSimplex_Result(smp.DPSimplex_Result, PostPredLoss):
#     def prediction(self):
#         predicted = np.empty((self.nSamp, self.nDat, self.nCol))
#         for s in range(self.nSamp):
#             zeta = self.samples.zeta[s][self.samples.delta[s]]
#             predicted[s] = gamma.rvs(a = zeta) + epsilon
#         return predicted
#
#     def __init__(self, path):
#         super().__init__(path)
#         self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
#         return
#
# class MPG_Result(mpg.MPGResult, PostPredLoss):
#     def prediction(self):
#         predicted = np.empty((self.nSamp, self.nDat, self.nCol))
#         for s in range(self.nSamp):
#             alpha = self.samples.alpha[s][self.samples.delta[s]]
#             beta  = self.samples.beta[s][self.samples.delta[s]]
#             predicted[s] = gamma.rvs(a = alpha, scale = 1/beta) + epsilon
#         return predicted
#
#     def __init__(self, path):
#         super().__init__(path)
#         self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
#         return
#
# class DPMPG_Result(dpmpg.ResultDPMPG, PostPredLoss):
#     def prediction(self):
#         predicted = np.empty((self.nSamp, self.nDat, self.nCol))
#         for s in range(self.nSamp):
#             alpha = self.samples.alpha[s][self.samples.delta[s]]
#             beta  = self.samples.beta[s][self.samples.delta[s]]
#             predicted[s] = gamma.rvs(a = alpha, scale = 1 / beta) + epsilon
#         return predicted
#
#     def __init__(self, path):
#         super().__init__(path)
#         self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
#         return
#
# class DPMRPG_Result(dpmrpg.ResultDPMPG, PostPredLoss):
#     def prediction(self):
#         predicted = np.empty((self.nSamp, self.nDat, self.nCol))
#         for s in range(self.nSamp):
#             zeta = self.samples.zeta[s][self.samples.delta[s]]
#             predicted[s] = gamma.rvs(a = zeta) + epsilon
#         return predicted
#
#     def __init__(self, path):
#         super().__init__(path)
#         self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
#         return
#
# class DPPGLN_Result(dppgln.DPMPG_Result, PostPredLoss):
#     def prediction(self):
#         predicted = np.empty((self.nSamp, self.nDat, self.nCol))
#         for s in range(self.nSamp):
#             alpha = self.samples.alpha[s][self.samples.delta[s]]
#             beta  = self.samples.beta[s][self.samples.delta[s]]
#             predicted[s] = gamma.rvs(a = alpha, scale = 1 / beta) + epsilon
#         return predicted
#
#     def __init__(self, path):
#         super().__init__(path)
#         self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
#         return
#
# Result = {
#     'fmix'   : FMIX_Result,
#     'mpg'    : MPG_Result,
#     'dpmpg'  : DPMPG_Result,
#     'dpmrpg' : DPMRPG_Result,
#     'dppgln' : DPPGLN_Result,
#     'dpmix'  : DPSimplex_Result,
#     }

Result = {
    'mpg'     : MPG_Result,
    'mprg'    : MPRG_Result,
    'md'      : MD_Result,
    'mgd'     : MGD_Result,
    'dppg'    : DPPG_Result,
    'dpprg'   : DPPRG_Result,
    'dpd'     : DPD_Result,
    'dpgd'    : DPGD_Result,
    'mpgln'   : MPGLN_Result,
    'mprgln'  : MPRGLN_Result,
    'mdln'    : MDLN_Result,
    'mgdln'   : MGDLN_Result,
    'dppgln'  : DPPGLN_Result,
    'dpprgln' : DPPRGLN_Result,
    'dpdln'   : DPDLN_Result,
    'dpgdln'  : DPGDLN_Result,
    'dppn'    : DPPN_Result,
    'mpn'     : MPN_Result,
    }

if __name__ == '__main__':
    base_path = './output'
    #model_types = ['fmix', 'dpmix', 'dpmrpg', 'dpmpg', 'dppgln', 'mpg']
    # model_types = [
    #         'md',   'dpd',   'mgd',   'dpgd',   'mprg',   'dpprg',   'mpg',   'dppg',
    #         'mdln', 'dpdln', 'mgdln', 'dpgdln', 'mprgln', 'dpprgln', 'mpgln', 'dppgln',
    #         ]
    model_types = ['dppn']

    models = []
    for model_type in model_types:
        mm = glob.glob(os.path.join(base_path, model_type, 'results_*.db'))
        for m in mm:
            models.append((model_type, m))

    # model = models[0]
    # result = Result[model[0]](model[1])
    ppl_slots = 'type name PPL_L1 PPL_L2 PPL_Linf ES_Linf'
    # PostPredLossResult = namedtuple('PostPredLossResult', 'type name PPL_L1 PPL_L2 PPL_Linf ES_L1 ES_L2 ES_Linf')
    PostPredLossResult = namedtuple('PostPredLossResult', ppl_slots)
    postpredlossresults = []
    for model in models:
        print('Processing {}-{}'.format(*model))
        result = Result[model[0]](model[1])
        postpredlossresults.append(
            PostPredLossResult(
                model[0],
                os.path.splitext(os.path.split(model[1])[1])[0],
                result.posterior_predictive_loss_L1(),
                result.posterior_predictive_loss_L2(),
                result.posterior_predictive_loss_Linf(),
                # result.energy_score_L1(),
                # result.energy_score_L2(),
                result.energy_score_Linf(),
                )
            )
        del(result)

    df = pd.DataFrame(
        postpredlossresults,
        # columns = ('type','name','PPL_L1','PPL_L2','PPL_Linf','ES_L1','ES_L2','ES_Linf'),
        columns = ('type','name','PPL_L1','PPL_L2','PPL_Linf','ES_Linf'),
        )
    # df.to_csv('./output/post_pred_loss_results_abbv.csv', index = False)

# EOF
