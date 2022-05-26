from re import M
from numpy.random import choice, gamma, beta, uniform, normal
from collections import namedtuple
from itertools import repeat, chain
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from math import log
from scipy.special import gammaln, betaln

import cUtility as cu
from samplers import DirichletProcessSampler
from cProjgamma import sample_alpha_1_mh_summary, sample_alpha_k_mh_summary
from cUtility import diriproc_cluster_sampler
from data import euclidean_to_angular, euclidean_to_hypercube, euclidean_to_simplex, MixedDataBase
from projgamma import GammaPrior
from model_cdppprg import logd_CDM_mx_ma, logd_CDM_mx_sa, logd_CDM_paired, logd_loggamma_mx
from model_sdpppg import dprodgamma_log_my_mt

from multiprocessing import Pool
from energy import limit_cpu

def update_zeta_j_cat(curr_zeta, Ws, alpha, beta, catmat):
    """ Update routine for zeta on categorical/multinomial data """
    curr_log_zeta = np.log(curr_zeta)
    prop_log_zeta = curr_log_zeta.copy()
    offset = normal(scale = 0.3, size = curr_zeta.shape)
    lunifs = np.log(uniform(size = curr_zeta.shape))
    logp = np.zeros(2)
    for i in range(curr_zeta.shape[0]):
        prop_log_zeta[i] += offset[i]
        logp += logd_CDM_mx_ma(
            Ws, 
            np.exp(np.vstack((curr_log_zeta, prop_log_zeta))), 
            catmat,
            ).sum(axis = 0)
        logp += logd_loggamma_mx(
            np.vstack((curr_log_zeta[i], prop_log_zeta[i])), 
            alpha[i], beta[i],
            ).ravel()
        if lunifs[i] < logp[1] - logp[0]:
            curr_log_zeta[i] = prop_log_zeta[i]
        else:
            prop_log_zeta[i] = curr_log_zeta[i]
        logp[:] = 0.
    return np.exp(curr_log_zeta)

def update_zeta_j_sph(curr_zeta, n, sY, slY, alpha, beta):
    """ Update routine for zeta on spherical data """
    prop_zeta = np.empty(curr_zeta.shape)
    for l in range(curr_zeta.shape[0]):
        prop_zeta[l] = sample_alpha_1_mh_summary(
            curr_zeta[l], n, sY[l], slY[l], alpha[l], beta[l]
            )
    return prop_zeta

def update_zeta_j_wrapper(args):
    # parse arguments
    curr_zeta_j, nj, sYj, slYj, Ws, alpha, beta, ncol, catmat = args
    prop_zeta_j = np.empty(curr_zeta_j.shape)
    prop_zeta_j[:ncol] = update_zeta_j_sph(
        curr_zeta_j[:ncol], nj, sYj, slYj, alpha[:ncol], beta[:ncol]
        )
    prop_zeta_j[ncol:] = update_zeta_j_cat(
        curr_zeta_j[ncol:], Ws, alpha[ncol:], beta[ncol:], catmat
        )
    return prop_zeta_j

def sample_gamma_shape_wrapper(args):
    return sample_alpha_k_mh_summary(*args)

Prior = namedtuple('Prior', 'eta alpha beta')

class Samples(object):
    zeta  = None
    alpha = None
    beta  = None
    delta = None
    r     = None
    eta   = None

    def __init__(self, nSamp, nDat, nCol, nCat, nCats):
        """
        nCol: number of 
        nCat: number of categorical columns
        nCats: number of categorical variables        
        """
        self.zeta  = [None] * (nSamp + 1)
        self.alpha = np.empty((nSamp + 1, nCol + nCat))
        self.beta  = np.empty((nSamp + 1, nCol + nCat))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        return

class Chain(DirichletProcessSampler):
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter]
    @property
    def curr_beta(self):
        return self.samples.beta[self.curr_iter]
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter]
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter]
    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter]

    def clean_delta_zeta(self, delta, zeta):
        """
        Find populated clusters, re-index them, 
        keep only parameters associated with extant clusters
        ---
        inputs:
            delta : cluster indicator vector (n)
            zeta  : cluster parameter matrix (J x d)
        outputs:
            delta : cluster indicator vector (n)
            zeta  : cluster parameter matrix (J* x d)
        """
        # Find populated clusters, re-index them
        keep, delta[:] = np.unique(delta, return_inverse = True)
        # return new indices, cluster parameters associated with populated clusters
        return delta, zeta[keep]

    def sample_zeta_new(self, alpha, beta, m):
        return gamma(shape = alpha, scale = 1 / beta, size = (m, self.nCol + self.nCat))

    def sample_alpha(self, zeta, curr_alpha):
        n = zeta.shape[0]
        zs = zeta.sum(axis = 0)
        lzs = np.log(zeta).sum(axis = 0)
        args = zip(
            curr_alpha, repeat(n), zs, lzs,
            repeat(self.priors.alpha.a), repeat(self.priors.alpha.b),
            repeat(self.priors.beta.a), repeat(self.priors.beta.b),
            )
        res = map(sample_gamma_shape_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        n  = zeta.shape[0]
        zs = zeta.sum(axis = 0)
        As = n * alpha + self.priors.beta.a
        Bs = zs + self.priors.beta.b
        beta = gamma(shape = As, scale = 1/Bs)
        return beta

    def sample_r(self, delta, zeta):
        As = np.einsum('il->i', zeta[delta].T[:self.nCol].T)
        Bs = np.einsum('il->i', self.data.Yp)
        return gamma(shape = As, scale = 1/Bs)

    def sample_eta(self, curr_eta, delta):
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + delta.max() + 1
        bb = self.priors.beta.b - log(g)
        eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_zeta(self, curr_zeta, r, delta, alpha, beta):
        nclust = delta.max() + 1
        dmat = delta[:,None] == np.arange(nclust) # n x J

        Y  = r[:, None] * self.data.Yp
        n = dmat.sum(axis = 0)
        Ysv = (Y.T @ dmat).T
        lYsv = (np.log(Y).T @ dmat).T
        Ws = [self.data.W[delta == j] for j in range(nclust)]
        # curr_zeta_j, nj, sYj, slYj, Ws, alpha, beta, ncol, catmat
        args = zip(
            curr_zeta, n, Ysv, lYsv, Ws,
            repeat(alpha), repeat(beta), 
            repeat(self.nCol), repeat(self.CatMat),
            )
        res = self.pool.map(update_zeta_j_wrapper, args)
        # res = map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def cluster_log_likelihood(self, r, zeta):
        out = np.zeros((self.nDat, self.max_clust_count))
        # out += dprojresgamma_log_my_mt(self.data.Yp, zeta.T[:self.nCol].T)
        out += dprodgamma_log_my_mt(
            r[:, None] * self.data.Yp, 
            zeta.T[:self.nCol].T, 
            self.sigma_placeholder,
            )
        out += logd_CDM_mx_ma(self.data.W, zeta.T[self.nCol:].T, self.CatMat)
        return out

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol, self.nCat, self.nCats)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.zeta[0] = gamma(
                shape = 2., scale = 2., 
                size = (self.max_clust_count - 30, self.nCol + self.nCat),
                )
        self.samples.eta[0] = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 30, size = self.nDat)
        self.samples.delta[0][-1] = np.arange(self.max_clust_count - 30)[-1]
        self.samples.r[0] = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0],
                )
        self.curr_iter = 0
        self.sigma_placeholder = np.ones((self.max_clust_count, self.nCol))
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy();  m = self.max_clust_count - (delta.max() + 1)
        alpha = self.curr_alpha
        beta  = self.curr_beta
        zeta  = np.vstack((self.curr_zeta, self.sample_zeta_new(alpha, beta, m)))
        eta   = self.curr_eta
        r     = self.curr_r

        self.curr_iter += 1
        # generate log-likelihood, uniforms to inverse-cdf sample cluster indices
        log_likelihood = self.cluster_log_likelihood(r, zeta)
        unifs = uniform(size = self.nDat)
        # Sample cluster indices
        delta = diriproc_cluster_sampler(delta, log_likelihood, unifs, eta)
        # clean indices (clear out dropped/unused clusters, and re-index)
        delta, zeta = self.clean_delta_zeta(delta, zeta)
        self.samples.delta[self.curr_iter] = delta
        # do rest of sampling
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_r, self.curr_delta, alpha, beta,
                )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter]  = self.sample_beta(self.curr_zeta, self.curr_alpha)
        self.samples.eta[self.curr_iter]   = self.sample_eta(eta, self.curr_delta)
        return

    def write_to_disk(self, path, nBurn, nThin = 1):
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.mkdir(folder)
        if os.path.exists(path):
            os.remove(path)

        zetas  = np.vstack([
            np.hstack((np.ones((zeta.shape[0], 1)) * i, zeta))
            for i, zeta in enumerate(self.samples.zeta[nBurn :: nThin])
            ])
        alphas = self.samples.alpha[nBurn :: nThin]
        betas  = self.samples.beta[nBurn :: nThin]
        deltas = self.samples.delta[nBurn :: nThin]
        rs     = self.samples.r[nBurn :: nThin]
        etas   = self.samples.eta[nBurn :: nThin]

        out = {
            'zetas'  : zetas,
            'alphas' : alphas,
            'betas'  : betas,
            'rs'     : rs,
            'deltas' : deltas,
            'etas'   : etas,
            'nCol'   : self.nCol,
            'nDat'   : self.nDat,
            'nCat'   : self.nCat,
            'cats'   : self.data.Cats,
            'V'      : self.data.V,
            'W'      : self.data.W,
            }
        
        try:
            out['Y'] = self.data.Y
        except AttributeError:
            pass
        
        with open(path, 'wb') as file:
            pickle.dump(out, file)

        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        self.data.Yp[self.data.Yp <= 1e-6] = 1e-6
        return
    
    def categorical_considerations(self):
        """ Builds the CatMat """
        cats = np.hstack(list(np.ones(ncat) * i for i, ncat in enumerate(self.data.Cats)))
        self.CatMat = (cats[:, None] == np.arange(len(self.data.Cats))).T
        return
    
    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_alpha = GammaPrior(0.5, 0.5),
            prior_beta  = GammaPrior(0.5, 0.5),
            p           = 10,
            max_clust_count = 300,
            ):
        assert type(data) is MixedData
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCat = self.data.nCat
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.nCats = self.data.Cats.shape[0]
        self.priors = Prior(prior_eta, prior_alpha, prior_beta)
        self.set_projection()
        self.categorical_considerations()
        self.pool = Pool(processes = 8, initializer = limit_cpu())
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = np.bincount(self.samples.delta[s], minlength = int(dmax + 1 + m))
            ljs = njs + (njs == 0) * self.samples.eta[s] / m
            new_zetas = gamma(
                shape = self.samples.alpha[s],
                scale = 1. / self.samples.beta[s],
                size = (m, self.nCol + self.nCat),
                )
            prob = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
            new_gammas.append(gamma(shape = zeta))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1, m = 10):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample, m)
        # hypercube transformation for real variates
        hypcube = euclidean_to_hypercube(gammas[:,:self.nCol])
        # simplex transformation for categ variates
        simplex_reverse = []
        indices = list(np.arange(self.nCol + self.nCat))
        # Foe each category, last first
        for i in list(range(self.cats.shape[0]))[::-1]:
            # identify the ending index (+1 to include boundary)
            cat_length = self.cats[i]
            cat_end = indices.pop() + 1
            # identify starting index
            for _ in range(cat_length - 1):
                cat_start = indices.pop()
            # transform gamma variates to simplex
            simplex_reverse.append(euclidean_to_simplex(gammas[:,cat_start:cat_end]))
        # stack hypercube and categorical variables side by side.
        return np.hstack([hypcube] + simplex_reverse[::-1])

    def generate_posterior_predictive_angular(self, n_per_sample = 1, m = 10):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample, m)
        return euclidean_to_angular(hyp)

    def write_posterior_predictive(self, path, n_per_sample = 1):
        colnames_y = ['Y_{}'.format(i) for i in range(self.nCol)]
        colnames_p = [
            ['p_{}_{}'.format(i,j) for j in range(catlength)]
            for i, catlength in enumerate(self.cats)
            ]
        colnames_p = list(chain(*colnames_p))

        thetas = pd.DataFrame(
                self.generate_posterior_predictive_hypercube(n_per_sample),
                # self.generate_posterior_predictive_angular(n_per_sample),
                #columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
                columns = colnames_y + colnames_p
                )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):        
        with open(path, 'rb') as file:
            out = pickle.load(file)
        
        deltas = out['deltas']
        etas   = out['etas']
        zetas  = out['zetas']
        alphas = out['alphas']
        betas  = out['betas']
        rs     = out['rs']
        cats   = out['cats']
        
        self.data = MixedDataBase(out['V'], out['W'], out['cats'])
        self.nSamp  = deltas.shape[0]
        self.nDat   = deltas.shape[1]
        self.nCat   = self.data.nCat
        self.nCol   = self.data.nCol
        self.nCats  = cats.shape[0]
        self.cats   = cats
        
        if 'Y' in out.keys():
            self.data.fill_outcome(out['Y'])
        
        self.samples       = Samples(self.nSamp, self.nDat, self.nCol, self.nCat, self.nCats)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.r     = rs
        return

    def __init__(self, path):
        self.load_data(path)
        return

def argparser():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('in_data_path')
    p.add_argument('out_path')
    p.add_argument('cat_vars')
    p.add_argument('--in_outcome_path', default = False)
    p.add_argument('--decluster', default = 'False')
    p.add_argument('--quantile', default = 0.95)
    p.add_argument('--nSamp', default = 30000)
    p.add_argument('--nKeep', default = 10000)
    p.add_argument('--nThin', default = 10)
    p.add_argument('--eta_alpha', default = 2.)
    p.add_argument('--eta_beta', default = 1.)
    return p.parse_args()

class Heap(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        return

if __name__ == '__main__':
    from data import MixedData
    from projgamma import GammaPrior
    from pandas import read_csv
    import os

    p = argparser()
    # d = {
    #     'in_data_path'    : './ad/cardio/data.csv',
    #     'in_outcome_path' : './ad/cardio/outcome.csv',
    #     'out_path' : './ad/cardio/results_mdppprg.pkl',
    #     'cat_vars' : '[15,16,17,18,19,20,21,22,23,24]',
    #     'decluster' : 'False',
    #     'quantile' : 0.95,
    #     'nSamp' : 30000,
    #     'nKeep' : 10000,
    #     'nThin' : 10,
    #     'eta_alpha' : 2.,
    #     'eta_beta' : 1.,
    #     }
    # p = Heap(**d)

    raw = read_csv(p.in_data_path).values
    out = read_csv(p.in_outcome_path).values
    data = MixedData(
        raw, 
        cat_vars = np.array(eval(p.cat_vars), dtype = int), 
        decluster = eval(p.decluster), 
        quantile = p.quantile,
        outcome = out,
        )
    data.fill_outcome(out)
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    model.sample(p.nSamp)
    model.write_to_disk(p.out_path, p.nKeep, p.nThin)
    res = Result(p.out_path)
    # res.write_posterior_predictive('./test/postpred.csv')

# EOF
