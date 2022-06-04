from re import L
from numpy.random import choice, gamma, beta, uniform, normal
from collections import namedtuple
from itertools import repeat
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from math import log, exp
from scipy.special import gammaln
from multiprocessing import Pool
from contextlib import nullcontext

from energy import limit_cpu
from cUtility import diriproc_cluster_sampler, generate_indices
from samplers import DirichletProcessSampler
from cProjgamma import sample_alpha_k_mh_summary, sample_alpha_1_mh_summary
from projgamma import GammaPrior, logd_loggamma_mx_st, logd_cumdirmultinom_mx_ma
from data import euclidean_to_angular, euclidean_to_hypercube, Data


# def logd_CDM_mx_sa(aW, vAlpha, sphere_mat):
#     sa = np.einsum('d,cd->c', vAlpha, sphere_mat)
#     sw = np.einsum('nd,cd->nc', aW, sphere_mat)
#     logd = np.zeros(aW.shape[0])
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         logd += gammaln(sa).sum()
#         logd += np.einsum('nc->n', gammaln(sw + 1))
#         logd -= np.einsum('nc->n', gammaln(sw + sa[None,:]))
#         logd += np.einsum('nd->n', gammaln(aW + vAlpha[None,:]))
#         logd -= gammaln(vAlpha).sum()
#         logd -= np.einsum('nd->n', gammaln(aW + 1))
#     np.nan_to_num(logd, False, -np.inf)
#     return logd

# def pt_logd_CDM_mx_sa(aW, aAlpha, sphere_mat):
#     sa = np.einsum('td,cd->tc', aAlpha, sphere_mat) # (t,c)
#     sw = np.einsum('nd,cd->nc', aW, sphere_mat)     # (n,c)
#     logd = np.zeros((aAlpha.shape[0], aW.shape[0])) # (t,n)
#     with np.errstate(divide = 'ignore',  invalid = 'ignore'):
#         logd += np.einsum('', gammaln(sa))[:,None] # (t,)
#         logd += np.einsum('nc->n', gammaln(sw + 1))[None,:] # (,n) 
#         logd -= np.einsum('tnc->tn', gammaln(sw[None,:,:] + sa[:,None,:])) # (t,n)
#         logd += np.einsum('tnd->tn', gammaln(aW[None,:,:] + aAlpha[:,None,:])) # (t,n)
#         logd -= np.einsum('td,t', gammaln(aAlpha))[:,None] # (t,)
#         logd -= np.einsum('nd->n', gammaln(aW + 1))[None,:]
#     np.nan_to_num(logd, False, -np.inf)
#     return logd

# def logd_CDM_mx_ma(aW, aAlpha, sphere_mat):
#     """
#     Log-density of concatenated Dirichlet-Multinomial distribution
#     ---
#     Inputs: 
#         aW         (n x d)
#         aAlpha     (j x d)
#         sphere_mat (c x d)
#     Output:
#         logd (n x j)
#     """
#     sa = np.einsum('jd,cd->jc', aAlpha, sphere_mat)
#     sw = np.einsum('nd,cd->nc', aW, sphere_mat)
#     logd = np.zeros((aW.shape[0], aAlpha.shape[0]))
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         logd += np.einsum('jc->j', gammaln(sa))[None,:]
#         logd += np.einsum('nc->n', gammaln(sw + 1))[:,None]
#         logd -= np.einsum('njc->nj', gammaln(sw[:,None,:] + sa[None,:,:]))
#         logd += np.einsum('njd->nj', gammaln(aW[:,None,:] + aAlpha[None,:,:]))
#         logd -= np.einsum('jd->j', gammaln(aAlpha))[None,:]
#         logd -= np.einsum('nd->n', gammaln(aW + 1))[:,None]
#     np.nan_to_num(logd, False, -np.inf)
#     return logd

# def pt_logd_CDM_mx_ma(aW, aAlpha, sphere_mat):
#     """
#     inputs:
#         aW:         (n,d)
#         aAlpha:     (t,j,d)
#         sphere_mat: (c,d)
#     output:
#         logd:       (n,t,j)
#     """
#     sa = np.einsum('tjd,cd->tjc', aAlpha, sphere_mat)
#     sw = np.einsum('nd,cd->nc', aW, sphere_mat)
#     logd = np.zeros((aW.shape[0], aAlpha.shape[0], aAlpha.shape[1]))
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         logd += np.einsum('tjc->tj', gammaln(sa))[None,:,:]
#         logd += np.einsum('nc->n', gammaln(sw + 1))[:,None,None]
#         logd -= np.einsum('tnjc->ntj', gammaln(sw[None,:,None,:] + sa[:,None,:,:])) # (n,c)+(tjc)->(tnjc)
#         logd += np.einsum('tnjd->ntj',gammaln(aW[None,:,None,:] + aAlpha[:,None,:,:])) # (nd)+(tjd)->(tnjd)
#         logd -= np.einsum('tjd->tj', gammaln(aAlpha))[None,:,:]
#         logd -= np.einsum('tnd->nt', gammaln(aW + 1))[:,:,None]
#     np.nan_to_num(logd, False, -np.inf)
#     return logd

# def logd_CDM_paired(aW, aAlpha, sphere_mat):
#     sa = np.einsum('nd,cd->nc', aAlpha, sphere_mat)
#     sw = np.einsum('nd,cd->nc', aW, sphere_mat)
#     logd = np.zeros(aW.shape[0])
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         logd += np.einsum('nc->n', gammaln(sa))
#         logd += np.einsum('nc->n', gammaln(sw + 1))
#         logd -= np.einsum('nc->n', gammaln(sw + sa))
#         logd += np.einsum('nd->n', gammaln(aW + aAlpha))
#         logd -= np.einsum('nd->n', gammaln(aAlpha))
#         logd -= np.einsum('nd->n', gammaln(aW + 1))
#     np.nan_to_num(logd, False, -np.inf)
#     return logd

# def pt_logd_CDM_paired(aW, aAlpha, sphere_mat):
#     """
#     returns log-likelihood per temperature
#     inputs:
#         aW         : (n,d)
#         aAlpha     : (t,n,d)
#         sphere_mat : (c,d)
#     outputs:
#         logd       : (t,n)
#     """
#     sa = np.einsum('tnd,cd->tnc', aAlpha, sphere_mat)
#     sw = np.einsum('nd,cd->nc')
#     logd = np.zeros((aAlpha.shape[0], aW.shape[0]))
#     # with np.errstate(divide = 'ignore', invalid = 'ignore'):
#     with nullcontext():
#         logd += np.einsum('tnc->tn', gammaln(sa))
#         logd += np.einsum('nc->n', gammaln(sw + 1))[None,:]
#         logd -= np.einsum('tnc->tn', gammaln(sw[None,:,:] + sa))
#         logd += np.einsum('tnd->tn', gammaln(aW[None,:,:] + aAlpha))
#         logd -= np.einsum('tnd->tn', gammaln(aAlpha))
#         logd -= np.einsum('nd->n', gammaln(aW + 1))[None,:]
#     # np.nan_to_num(logd, False, -np.inf)
#     return logd

# def logd_loggamma_single(x, a, b):
#     logd = 0.
#     logd += a * log(b)
#     logd -= gammaln(a)
#     logd += a * x
#     logd -= b * exp(x)
#     return logd

# def logd_loggamma_mx(x, a, b):
#     logd = np.zeros(x.shape[0])
#     logd += a * np.log(b)
#     logd -= gammaln(a)
#     logd += a * x
#     logd -= b * np.exp(x)
#     return logd

# def pt_logd_loggamma(x, a, b):
#     """
#     Log-density of log-gamma distribution at multiple temperatures
#     inputs:
#         x : (t,j,d) # (m,d)
#         a : (t,d)   # (m,d)
#         b : (t,d)   # (m,d)
#     output:
#         out : (t,j) # (m)
#     """
#     logd = np.empty((x.shape[0]))
#     logd += np.einsum('md,md->m', a, np.log(b))
#     logd -= np.einsum('md->m', gammaln(a))
#     logd += np.einsum('md,md->m', a, x)
#     logd -= np.einsum('md,md->m', b, np.exp(x))
#     # logd = np.empty((x.shape[0], x.shape[1]))
#     # logd += np.einsum('td,td->t', a, np.log(b))
#     # logd -= np.einsum('td->t', gammaln(a))
#     # logd += np.einsum('tjd,td->tj', a, x)
#     # logd -= np.einsum('td,tjd->tj', b, np.exp(x))
#     return logd

# def update_zeta_j_wrapper_old(args):
#     """
#     Args:
#         args (tuple):
#             curr_zeta (d)
#             Ws        (n x d)
#             alpha     (d)
#             beta      (d)
#             spheres   (tuple of integer arrays)
#     """
#     curr_zeta, Ws, alpha, beta, sphere_mat = args
#     curr_log_zeta = np.log(curr_zeta)
#     prop_log_zeta = curr_log_zeta + normal(scale = 0.3, size = curr_zeta.shape)
#     eval_log_zeta = curr_log_zeta.copy()
#     lunifs = np.log(uniform(size = curr_zeta.shape))
#     logp = np.zeros(2)
#     for i in range(curr_zeta.shape[0]):
#         logp[0] += logd_cumdirmultinom_mx_sa(Ws, np.exp(eval_log_zeta), sphere_mat).sum()
#         logp[0] += logd_loggamma_sx(eval_log_zeta[i], alpha[i], beta[i])
#         eval_log_zeta[i] = prop_log_zeta[i]
#         logp[1] += logd_cumdirmultinom_mx_sa(Ws, np.exp(eval_log_zeta), sphere_mat).sum()
#         logp[1] += logd_loggamma_sx(eval_log_zeta[i], alpha[i], beta[i])
#         if lunifs[i] < logp[1] - logp[0]:
#             pass
#         else:
#             eval_log_zeta[i] = curr_log_zeta[i]
#         logp[:] = 0.
#     return np.exp(eval_log_zeta)

def update_zeta_j_wrapper(args):
    """
    Args:
        args (tuple):
            curr_zeta  (d)
            Ws         (n x d)
            alpha      (d)
            beta       (d)
            sphere_mat (tuple of integer arrays)
    """
    curr_zeta, Ws, alpha, beta, sphere_mat = args
    curr_log_zeta = np.log(curr_zeta)
    prop_log_zeta = curr_log_zeta.copy()
    offset = normal(scale = 0.3, size = curr_zeta.shape)
    lunifs = np.log(uniform(size = curr_zeta.shape))
    logp = np.zeros(2)
    for i in range(curr_zeta.shape[0]):
        prop_log_zeta[i] += offset[i]
        logp += logd_cumdirmultinom_mx_ma(Ws, np.exp(np.vstack((curr_log_zeta, prop_log_zeta))), sphere_mat).sum(axis = 0)
        logp += logd_loggamma_mx_st(np.hstack((curr_log_zeta[i], prop_log_zeta[i])), alpha[i], beta[i]).ravel()
        if lunifs[i] < logp[1] - logp[0]: # if accept, then set current to new value
            curr_log_zeta[i] = prop_log_zeta[i]
        else:                             # otherwise, set proposal to current
            prop_log_zeta[i] = curr_log_zeta[i]
        logp[:] = 0.
    return np.exp(curr_log_zeta)

# def dprodgamma_log_my_mt(aY, aAlpha, aBeta):
#     """
#     Product of Gammas log-density for multiple Y, multiple theta (not paired)
#     ----
#     aY     : array of Y     (n x d)
#     aAlpha : array of alpha (J x d)
#     aBeta  : array of beta  (J x d)
#     ----
#     return : array of ld    (n x J)
#     """
#     out = np.zeros((aY.shape[0], aAlpha.shape[0]))
#     out += np.einsum('jd,jd->j', aAlpha, np.log(aBeta)).reshape(1,-1) # beta^alpha
#     out -= np.einsum('jd->j', gammaln(aAlpha)).reshape(1,-1)          # gamma(alpha)
#     out += np.einsum('jd,nd->nj', aAlpha - 1, np.log(aY))             # y^(alpha - 1)
#     out -= np.einsum('jd,nd->nj', aBeta, aY)                          # e^(- beta y)
#     return out

def sample_gamma_shape_wrapper(args):
    return sample_alpha_k_mh_summary(*args)

Prior = namedtuple('Prior', 'eta alpha beta')

class Samples(object):
    zeta  = None
    alpha = None
    beta  = None
    delta = None
    eta   = None

    def __init__(self, nSamp, nDat, nCol):
        self.zeta  = [None] * (nSamp + 1)
        self.alpha = np.empty((nSamp + 1, nCol))
        self.beta  = np.empty((nSamp + 1, nCol))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
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
    def curr_delta(self):
        return self.samples.delta[self.curr_iter]
    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter]

    def clean_delta_zeta(self, delta, zeta):
        """
        delta : cluster indicator vector (n)
        zeta  : cluster parameter matrix (J* x d)
        """
        # which clusters are populated
        # keep = np.bincounts(delta) > 0 
        # reindex those clusters
        keep, delta[:] = np.unique(delta, return_inverse = True)
        # return new indices, cluster parameters associated with populated clusters
        return delta, zeta[keep]

    def sample_zeta_new(self, alpha, beta, m):
        return gamma(shape = alpha, scale = 1/beta, size = (m, self.nCat))

    def sample_alpha(self, zeta, curr_alpha):
        n    = zeta.shape[0]
        zs   = zeta.sum(axis = 0)
        lzs  = np.log(zeta).sum(axis = 0)
        args = zip(
            curr_alpha, repeat(n), zs, lzs,
            repeat(self.priors.alpha.a), repeat(self.priors.alpha.b),
            repeat(self.priors.beta.a), repeat(self.priors.beta.b),
            )
        res = map(sample_gamma_shape_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        n = zeta.shape[0]
        zs = zeta.sum(axis = 0)
        As = n * alpha + self.priors.beta.a
        Bs = zs + self.priors.beta.b
        return gamma(shape = As, scale = 1 / Bs)
    
    def sample_eta(self, curr_eta, delta):
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + delta.max() + 1
        bb = self.priors.beta.b - log(g)
        eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_zeta(self, curr_zeta, delta, alpha, beta):
        Ws = [self.data.W[delta == i] for i in range(delta.max() + 1)]
        # curr_zeta, Ws, alpha, beta, spheres = args
        args = zip(curr_zeta, Ws, repeat(alpha), repeat(beta), repeat(self.sphere_mat))
        res = self.pool.map(update_zeta_j_wrapper, args)
        # res = map(update_zeta_j_wrapper, args)
        return np.array(list(res))
    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCat)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 30, self.nCat))
        self.samples.eta[0] = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 30, size = self.nDat)
        self.samples.delta[0][-1] = np.arange(self.max_clust_count - 30)[-1]
        self.curr_iter = 0
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy();  m = self.max_clust_count - (delta.max() + 1)
        alpha = self.curr_alpha
        beta  = self.curr_beta
        zeta  = np.vstack((self.curr_zeta, self.sample_zeta_new(alpha, beta, m)))
        eta   = self.curr_eta

        self.curr_iter += 1
        # calculate log-likelihood under extant and candidate clusters
        log_likelihood = logd_cumdirmultinom_mx_ma(self.data.W, zeta, self.sphere_mat)
        # pre-generate uniforms to inverse-cdf sample cluster indices
        unifs = uniform(size = self.nDat)
        # sample new cluster membership indicators)
        delta = diriproc_cluster_sampler(delta, log_likelihood, unifs, eta)
        # clean indices (clear out dropped clusters, unused candidate clusters, and re-index)
        delta, zeta = self.clean_delta_zeta(delta, zeta)
        self.samples.delta[self.curr_iter] = delta
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_delta, alpha, beta,
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
        etas   = self.samples.eta[nBurn :: nThin]

        out = {
            'zetas'  : zetas,
            'alphas' : alphas,
            'betas'  : betas,
            'deltas' : deltas,
            'etas'   : etas,
            'nCat'   : self.nCat,
            'nDat'   : self.nDat,
            'W'      : self.data.W,
            'cats'   : self.data.Cats,
            'spheres': self.data.spheres,
            }
        
        try:
            out['Y'] = self.data.Y
        except AttributeError:
            pass
        
        with open(path, 'wb') as file:
            pickle.dump(out, file)

        return

    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 1.),
            prior_alpha = GammaPrior(0.5, 0.5),
            prior_beta  = GammaPrior(2., 2.),
            p           = 10,
            max_clust_count = 300,
            ):
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCat = self.data.nCat
        self.nDat = self.data.nDat
        self.spheres = self.data.spheres
        self.sphere_mat = np.zeros((len(self.spheres), self.nCat))
        for i, sphere in enumerate(self.spheres):
            self.sphere_mat[i][sphere] = True
        self.priors = Prior(prior_eta, prior_alpha, prior_beta)
        self.pool = Pool(processes = 8, initializer =  limit_cpu)
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
                size = (m, self.nCat),
                )
            prob = ljs / ljs.sum()
            deltas = generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
            new_gammas.append(gamma(shape = zeta))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1, m = 10):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample, m)
        return euclidean_to_hypercube(gammas)

    def generate_posterior_predictive_angular(self, n_per_sample = 1, m = 10):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample, m)
        return euclidean_to_angular(hyp)

    def write_posterior_predictive(self, path, n_per_sample = 1):
        thetas = pd.DataFrame(
                self.generate_posterior_predictive_angular(n_per_sample),
                columns = ['theta_{}'.format(i) for i in range(1, self.nCat)],
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

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCat  = alphas.shape[1]

        self.data = Multinomial(out['W'], out['cats'])
        self.spheres = out['spheres']
        self.sphere_mat = np.zeros((len(self.spheres), self.nCat))
        for i, sphere in enumerate(self.spheres):
            self.sphere_mat[i][sphere] = True
        
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass
        
        self.samples       = Samples(self.nSamp, self.nDat, self.nCat)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.zeta  = [
            zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)
            ]
        return

    def __init__(self, path):
        self.load_data(path)
        return

def argparser():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('in_path')
    p.add_argument('out_path')
    p.add_argument('cats')
    p.add_argument('--nSamp', default = 20000)
    p.add_argument('--nKeep', default = 10000)
    p.add_argument('--nThin', default = 5)
    return p.parse_args()

if __name__ == '__main__':
    # p = argparser()

    class Heap(object):
        def __init__(self, **kwargs):
            self.__dict__.update(**kwargs)
            return

    d = {
        'in_path'  : './simulated/categorical/test22.csv',
        'out_path' : './simulated/categorical/results_test22.pkl',
        'cats'     : '[2,2]',
        'nSamp'    : 20000,
        'nKeep'    : 10000,
        'nThin'    : 5,
        }
    p = Heap(**d)

    from data import Multinomial
    from pandas import read_csv 
    import os
    raw = read_csv(p.in_path).values
    data = Multinomial(raw, np.array(eval(p.cats), dtype = int))
    model = Chain(data)
    model.sample(p.nSamp)
    model.write_to_disk(p.out_path, p.nKeep, p.nThin)

# EOF
