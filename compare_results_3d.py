# Compute Distance / Divergence between Distributions
import numpy as np
import pandas as pd
import os, glob
import sqlite3 as sql
import itertools as it
from scipy.stats import gamma, norm as normal
from scipy.linalg import cholesky
from random import sample
import plotly.express as px

import simplex as smp
import m_projgamma as mpg
import dp_projgamma as dpmpg
import dp_pgln as dppgln

import cUtility as cu

from data import Data

np.seterr(under = 'ignore')

class Plot3d(object):
    data = None
    png_name = 'simplex_{}_{}_{}.png'

    def plot_3d(self, path):
        colsets = list(it.combinations(range(8), 3))
        for colset in colsets:
            colnames = dict(zip(['a','b','c'], ['C_{}'.format(x) for x in colset]))
            ppc = pd.DataFrame(self.generate_posterior_predictive_simplex_3d(colset),
                                columns = [colnames[x] for x in ['a','b','c']])
            ppc['source'] = 'PostPred'
            emc = pd.DataFrame(self.data.Yl[:,colset],
                                columns = [colnames[x] for x in ['a','b','c']])
            emc['source'] = 'Empirical'
            out_df = pd.concat([emc, ppc], axis = 0)

            fig = px.scatter_ternary(out_df, color = 'source', **colnames)
            fig.write_image(os.path.join(path, self.png_name.format(*colset)))

            del(fig)
            del(emc)
            del(ppc)
        return

class FMIX_Result_3d(smp.FMIX_Result, Plot3d):
    def generate_posterior_predictive_simplex_3d(self, cols):
        keep_idx = np.array(sample(range(self.nSamp), self.nDat), dtype = int)

        pi  = self.samples.pi[keep_idx]
        eta = (self.samples.eta[keep_idx])[:,:,np.array(cols, dtype = int)]

        postpred = np.empty((self.nDat, len(cols)))
        for n in range(self.nDat):
            delta_new = np.random.choice(range(self.nMix), 1, p = pi[n])
            eta_new = eta[n, delta_new]
            #postpred[n] = np.apply_along_axis(
            #    lambda a: gamma.rvs(a = a), 1, eta_new,
            #    ).reshape(-1)
            postpred[n] = gamma.rvs(a = eta_new)
        np.nan_to_num(postpred, copy = False)
        postpred += 1e-20
        return (postpred.T / (postpred.sum(axis = 1))).T

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class MPG_Result_3d(mpg.MPGResult, Plot3d):
    def generate_posterior_predictive_simplex_3d(self, cols):
        keep_idx = np.array(sample(range(self.nSamp), self.nDat), dtype = int)

        eta = self.samples.eta[keep_idx]
        alpha = self.samples.alpha[keep_idx][:,:,np.array(cols, dtype = int)]
        beta = self.samples.beta[keep_idx][:,:,np.array(cols, dtype = int)]

        postpred = np.empty((self.nDat, len(cols)))
        for n in range(self.nDat):
            delta_new = np.random.choice(range(self.nMix), 1, p = eta[n])
            alpha_new = alpha[n, delta_new]
            beta_new = beta[n, delta_new]
            postpred[n] = gamma.rvs(a = alpha_new, scale = 1/beta_new)
        np.nan_to_num(postpred, copy = False)
        postpred += 1e-20
        return (postpred.T / postpred.sum(axis = 1)).T

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class DPMPG_Result_3d(dpmpg.ResultDPMPG, Plot3d):
    def generate_posterior_predictive_simplex_3d(self, cols, m = 20):
        keep_idx = np.array(sample(range(self.nSamp), self.nDat), dtype = int)

        delta = self.samples.delta[keep_idx]
        alpha = [self.samples.alpha[i][:,cols] for i in keep_idx]
        beta  = [self.samples.beta[i][:,cols] for i in keep_idx]
        eta   = self.samples.eta[keep_idx]
        alpha_shape = self.samples.alpha_shape[keep_idx][:,cols]
        if cols[0] == 0:
            # beta_shape = np.array(
            #     [1, self.samples.beta_shape[keep_idx][:,(np.array(cols[1:], dtype = int) - 1)]]
            #     )
            beta_shape = np.hstack((
                np.ones((alpha_shape.shape[0],1)),
                self.samples.beta_shape[keep_idx][:,(np.array(cols[1:], dtype = int) - 1)]
                ))
        else:
            beta_shape = self.samples.beta_shape[keep_idx][:,(np.array(cols, dtype = int) - 1)]

        postpred = np.empty((self.nDat, len(cols)))
        for n in range(self.nDat):
            dmax = delta[n].max()
            njs = cu.counter(delta[n], dmax + 1 + m)
            ljs = njs + (njs == 0) * eta[n] / m
            new_alphas = gamma.rvs(a = alpha_shape[n], size = (m, len(cols)))
            if cols[0] == 0:
                new_betas = np.hstack((
                    np.ones((m, 1)),
                    gamma.rvs(a = beta_shape[n,1:], size = (m, len(cols) - 1))
                    ))
            else:
                new_betas = gamma.rvs(a = beta_shape[n], size = (m, len(cols)))
            prob = ljs / ljs.sum()
            a = np.vstack((alpha[n], new_alphas))
            b = np.vstack((beta[n], new_betas))
            new_delta = np.random.choice(range(dmax + 1 + m), 1, p = prob)
            postpred[n] = gamma.rvs(a = a[new_delta], scale = 1/b[new_delta])
        np.nan_to_num(postpred, copy = False)
        return (postpred.T / postpred.sum(axis = 1)).T

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class DPPGLN_Result_3d(dppgln.DPMPG_Result, Plot3d):
    def generate_posterior_predictive_simplex_3d(self, cols, m = 20):
        keep_idx = np.array(sample(range(self.nSamp), self.nDat), dtype = int)

        delta = self.samples.delta[keep_idx]
        alpha = [self.samples.alpha[i][:,cols] for i in keep_idx]
        beta  = [self.samples.beta[i][:,cols] for i in keep_idx]
        eta   = self.samples.eta[keep_idx]
        mu    = self.samples.mu[keep_idx][:,cols]
        Sigma = self.samples.Sigma[keep_idx][:,cols][:,:,cols]

        postpred = np.empty((self.nDat, len(cols)))
        for n in range(self.nDat):
            dmax = delta[n].max()
            njs = cu.counter(delta[n], dmax + 1 + m)
            ljs = njs + (njs == 0) * eta[n] / m
            new_log_alpha = mu[n].reshape(1,-1) + \
                    (cholesky(Sigma[n]) @ normal.rvs(size = (len(cols), m))).T
            new_alpha = np.exp(new_log_alpha)
            if cols[0] == 0:
                new_beta = np.hstack((
                    np.ones((m, 1)),
                    gamma.rvs(a = 2., scale = 1/2., size = (m, len(cols) - 1))
                    ))
            else:
                new_beta = gamma.rvs(a = 2., scale = 1/2, size = (m, len(cols)))
            prob = ljs / ljs.sum()
            a = np.vstack((alpha[n], new_alpha))
            b = np.vstack((beta[n], new_beta))
            delta_new = np.random.choice(range(dmax + 1 + m), 1, p = prob)
            postpred[n] = gamma.rvs(a = a[delta_new], scale = 1 / b[delta_new])
        np.nan_to_num(postpred, copy = False)
        return (postpred.T / postpred.sum(axis = 1)).T

    def __init__(self, path):
        super().__init__(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

Result = {
    'fmix'   : FMIX_Result_3d,
    'mpg'    : MPG_Result_3d,
    'dpmpg'  : DPMPG_Result_3d,
    'dppgln' : DPPGLN_Result_3d,
    }

if __name__ == '__main__':
    base_path = './output'
    model_types = ['fmix','dpmpg','dppgln','mpg'] #,'dpmp']

    models = []
    for model_type in model_types:
        mm = glob.glob(os.path.join(base_path, model_type, 'results_*.db'))
        for m in mm:
            models.append((model_type, m))

    for model in models:
        result = Result[model[0]](model[1])
        out_path = os.path.splitext(model[1])[0] + '_plots_3d'
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass
        result.plot_3d(out_path)
        del(result)

# EOF
