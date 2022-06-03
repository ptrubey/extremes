from re import M, S
from numpy.random import choice, gamma, beta, uniform, normal
from numpy.linalg import cholesky
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
from model_cdppprg import logd_CDM_mx_ma, logd_CDM_mx_sa, logd_CDM_paired, logd_loggamma_mx, \
                        pt_logd_CDM_mx_ma, pt_logd_CDM_mx_sa, pt_logd_CDM_paired, pt_logd_loggamma
from model_sdpppg import dprodgamma_log_my_mt
from model_sdpppgln import bincount2D_vectorized, dgamma_log_my, cluster_covariance_mat, \
                        dprodgamma_log_paired_yt, dprojgamma_log_paired_yt
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

def category_matrix(cats):
    catvec = np.hstack(list(np.ones(ncat) * i for i, ncat in enumerate(cats)))
    CatMat = (catvec[:, None] == np.arange(len(cats))).T
    return CatMat

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
    @property
    def curr_cluster_count(self):
        return self.curr_delta[0].max() + 1
    def average_cluster_count(self, ns):
        acc = self.samples.delta[(ns//2):][:,0].max(axis = 1).mean() + 1
        return '{:.2f}'.format(acc)

    # Adaptive Metropolis Placeholders
    am_cov_c  = None
    am_cov_i  = None
    am_mean_c = None
    am_mean_i = None
    am_n_c    = None
    am_alpha  = None
    max_clust_count = None

    def sample_delta(self, r, delta, zeta, eta):
        Y = r[:,:,None] * self.data.Yp[None,:,:]
        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        cand_cluster_state = (curr_cluster_state == 0)
        log_likelihood = np.zeros((self.nDat, self.nTemp, self.max_clust_count))
        log_likelihood += dprodgamma_log_my_mt(Y, zeta[:,:,:self.nCol], self.sigma_ph1)
        log_likelihood += pt_logd_CDM_mx_ma(self.data.W, zeta[:,:,self.nCol:], self.sphere_mat)
        tidx = np.arange(self.nTemp)
        p = uniform(size = (self.nDat, self.nTemp))
        p += tidx[None,:]
        scratch = np.empty(curr_cluster_state.shape)
        for i in range(self.nDat):
            curr_cluster_state[tidx, delta.T[i]] -= 1
            scratch[:] = 0
            scratch += curr_cluster_state
            scratch += cand_cluster_state * (eta / cand_cluster_state.sum(axis = 1) + 1e-9)[None,:]
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                np.log(scratch, out = scratch)
            scratch += log_likelihood[i]
            np.nan_to_num(scratch, False, -np.inf)
            scratch -= scratch.max(axis = 1)[:,None]
            with np.errstate(under = 'ignore'):
                np.exp(scratch, out = scratch)
            np.cumsum(scratch, axis = 1, out = scratch)
            scratch /= scratch.T[-1][:,None]
            scratch += tidx[:,None]
            delta.T[i] = np.searchsorted(scratch.ravel(), p[i]) % self.max_clust_count
            curr_cluster_state[tidx, delta.T[i]] += 1
            cand_cluster_state[tidx, delta.T[i]] = False
        return
    
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
        # keep, delta[:] = np.unique(delta, return_inverse = True)
        for t in range(self.nTemp):
            keep, delta[t] = np.unique(delta[t], return_inverse = True)
            zeta[t][:keep.shape[0]] = zeta[t,keep]
        # return new indices, cluster parameters associated with populated clusters
        # return delta, zeta[keep]
        return

    def sample_zeta_new(self, alpha, beta):
        # return gamma(shape = alpha, scale = 1 / beta, size = (m, self.nCol + self.nCat))
        # out = np.empty((self.nTemp, self.max_clust_count, self.nCol + self.nCat))
        # np.einsum(
        #     'tzy,tjy->tjz', 
        #     np.triu(Sigma_chol),
        #     normal(size = out.shape),
        #     out = out,
        #     )
        # out += mu[:,None,:]
        # np.exp(out, out = out)
        out = gamma(
            shape = alpha[:,None,:], 
            scale = 1 / beta[:,None,:], 
            size = (self.nTemp, self.max_clust_count, self.nCol + self.nCat),
            )
        return out

    def am_covariance_matrices(self, delta, index):
        cluster_covariance_mat(
            self.am_cov_c, self.am_mean_c, self.am_n_c, delta,
            self.am_cov_i, self.am_mean_i, self.curr_iter, np.arange(self.temp),
            )
        return self.am_cov_c[index]

    def sample_zeta(self, zeta, delta, r, alpha, beta):
        """
        zeta      : (t x J x D)
        delta     : (t x n)
        r         : (t x n)
        mu        : (t x D)
        Sigma_cho : (t x D x D)
        Sigma_inv : (t x D x D)
        """
        Y = r[:,:,None] * self.data.Yp[None,:,:]
        lY = np.log(Y)
        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        cand_cluster_state = (curr_cluster_state == 0)
        delta_ind_mat = delta[:,:,None] == range(self.max_clust_count)

        idx = np.where(~cand_cluster_state)
        covs = self.am_covariance_matrices(delta, idx)
        lzcurr = np.log(zeta)
        lzcand = lzcurr.copy()
        lzcand[idx] += np.einsum('mpq,mq->mp', cholesky(covs), normal(size = (idx[0].shape[0], self.nCol)))
        zcand = np.exp(zeta)
        
        z1_shape = [self.nTemp, self.nDat, self.nCol]
        z2_shape = [self.nTemp, self.nDat, self.nCat]
        self.am_alpha[:] = -np.inf
        self.am_alpha[idx] = 0.
        self.am_alpha[idx] += dprojgamma_log_paired_yt(
            self.data.Yp,
            zcand[:,:,:self.nCol][self.temp_unravel, delta.ravel()].reshape(z1_shape),
            self.sigma_ph2,
            )
        self.am_alpha[idx] += logd_CDM_paired(
            self.data.W,
            zcand[:,:,self.nCol:][self.temp_unravel, delta.ravel()].reshape(z2_shape),
            self.CatMat,
            )
        self.am_alpha[idx] -= dprodgamma_log_paired_yt(
            self.data.Yp,
            zeta[:,:,:self.nCol][self.temp_unravel, delta.ravel()].reshape(z1_shape),
            self.sigma_ph2,
            )
        self.am_alpha[idx] -= logd_CDM_paired(
            self.data.W,
            zcand[:,:,self.nCol:][self.temp_unravel, delta.ravel()].reshape(z2_shape),
            self.CatMat,
            )
        self.am_alpha[idx] *= self.itl[idx[0]]
        self.am_alpha[idx] += pt_logd_loggamma(lzcand[idx], alpha[idx[0]], beta[idx[0]])
        self.am_alpha[idx] -= pt_logd_loggamma(lzcurr[idx], alpha[idx[0]], beta[idx[0]])
        keep = np.where(np.log(uniform(size = self.am_alpha.shape)) < self.am_alpha)
        zeta[keep] = np.exp(lzcand[keep])
        return zeta

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
        self.CatMat = category_matrix(self.data.Cats)
        # cats = np.hstack(list(np.ones(ncat) * i for i, ncat in enumerate(self.data.Cats)))
        # self.CatMat = (cats[:, None] == np.arange(len(self.data.Cats))).T
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

    def generate_posterior_predictive_spheres(self):
        rhos = self.generate_posterior_predictive_gammas() # (s,D)
        CatMat = category_matrix(self.data.Cats) # (C,d)
        shro = rhos[:,self.nCol:] @ CatMat.T # (s,C)
        nrho = np.einsum('sc,cd->sd', shro, CatMat) # (s,d)
        pis = rhos / nrho
        return pis

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

    def generate_conditional_posterior_predictive_radii(self):
        """ r | zeta, V ~ Gamma(r | sum(zeta), sum(V)) """
        # As = np.einsum('il->i', zeta[delta].T[:self.nCol].T)
        # Bs = np.einsum('il->i', self.data.Yp)
        shapes = np.array([
            zeta[delta]
            for delta, zeta
            in zip(self.samples.delta, self.samples.zeta)
            ]).sum(axis = 2)
        rates = self.data.V.sum(axis = 1)[None,:]
        rs = gamma(shape = shapes, scale = 1 / rates)
        return rs

    def generate_conditional_posterior_predictive_gammas(self):
        """ rho | zeta, delta + W ~ Gamma(rho | zeta[delta] + W) """
        zetas = np.array([
            zeta[delta]
            for delta, zeta 
            in zip(self.samples.delta, self.samples.zeta)
            ]) # (s,n,d)
        W = np.hstack((np.zeros((self.nDat, self.nCol)), self.data.W)) # (n,d)
        return gamma(shape = zetas + W[None,:,:])

    def generate_conditional_posterior_predictive_spheres(self):
        """ pi | zeta, delta = normalized rho
        currently discarding generated Y's, keeping latent pis
        """
        rhos = self.generate_conditional_posterior_predictive_gammas() # (s,n,D)
        CatMat = category_matrix(self.data.Cats) # (C,d)
        shro = rhos[:,:,self.nCol:] @ CatMat.T # (s,n,C)
        nrho = np.einsum('snc,cd->snd', shro, CatMat) # (s,n,d)
        pis = rhos / nrho
        return pis

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
    p.add_argument('--nSamp', default = 50000)
    p.add_argument('--nKeep', default = 20000)
    p.add_argument('--nThin', default = 30)
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
    #     'in_data_path'    : './ad/cover/data.csv',
    #     'in_outcome_path' : './ad/cover/outcome.csv',
    #     'out_path' : './ad/cover/results_mdppprg.pkl',
    #     'cat_vars' : '[9,10,11,12]',
    #     'decluster' : 'False',
    #     'quantile' : 0.998,
    #     'nSamp' : 50000,
    #     'nKeep' : 20000,
    #     'nThin' : 30,
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
        quantile = float(p.quantile),
        outcome = out,
        )
    data.fill_outcome(out)
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    model.sample(p.nSamp)
    model.write_to_disk(p.out_path, p.nKeep, p.nThin)
    res = Result(p.out_path)
    # res.write_posterior_predictive('./test/postpred.csv')

# EOF
