from projgamma import *
from scipy.stats import gamma, beta, gmean
from numpy.random import choice
from collections import namedtuple
# from concurrent.futures import ThreadPoolExecutor
import numpy as np

BNPPGPrior = namedtuple('BNPPGPrior', 'alpha beta eta')
Theta      = namedtuple('Theta','alpha beta')

def log_density_gamma_i(args):
    return logdprojgamma_pre_single(*args)
def update_alpha_wrapper(args):
    if args[0] == 0:
        return sample_alpha_1_mh(*args[1:])
    elif args[0] > 0:
        return sample_alpha_k_mh(*args[1:])
    else:
        raise ValueError('Something other than col index was passed!')
def update_beta_wrapper(args):
    return sample_beta_fc(*args)

class DPMPG(object):
    samples_alpha = []
    samples_beta = []
    samples_delta = None
    executor = None

    def clean_delta(self, deltas, alphas, betas, i):
        """ delta is a vector of cluster assignments.  If the delta vector has
        0 entries for one index, and > 0 entries for a higher index, then the
        higher indices are shuffled down by one.  This is kind of wasteful if it
        need happen several times... """
        assert (deltas.max() + 1 == alphas.shape[0])
        _delta = np.delete(deltas, i)
        _alpha = alphas[np.array([j for j in range(_delta.max() + 1) if j in set(_delta)])]
        _beta  = betas[np.array([j for j in range(_delta.max() + 1) if j in set(_delta)])]
        nj     = np.array([(_delta == j).sum() for j in range(_delta.max() + 2)], dtype = int)
        fz     = np.where(nj == 0)[0][0]
        while fz <= _delta.max():
            _delta[_delta > fz] = _delta[_delta > fz] - 1
            nj = np.array([(_delta == j).sum() for j in range(_delta.max() + 2)], dtype = int)
            fz = np.where(nj == 0)[0][0]
        return _delta, _alpha, _beta

    def sample_delta_i(self, deltas, alphas, betas, eta, i):
        # Clean the deltas, alphsas, and betas.  calculate the new max delta
        _delta, _alpha, _beta = self.clean_delta(deltas, alphas, betas, i)
        _dmax = _delta.max()
        # Compute the prior probabilities for the collapsed sampler.
        nj = np.array([(_delta == j).sum() for j in range(_dmax + 1 + self.m)])
        lj = nj + (nj == 0) * eta / self.m
        # Generate potential new clusters
        alpha_new, beta_new = self.sample_alpha_beta_new()
        alpha_stack = np.vstack(_alpha, alpha_new)
        beta_stack  = np.vstack(_beta, beta_new)
        assert (alpha_stack.shape[0] == lj.shape[0])
        # Calculate log-posteriors under each cluster
        args = zip(
            repeat(self.data.lcoss[i]),
            repeat(self.data.lsins[i]),
            repeat(self.data.Yl[i]),
            alpha_stack.tolist(),
            beta_stack.tolist(),
            )
        lps = self.pool.map(log_density_gamma_i, args)
        unnormalized = exp(lps) * lj
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(_dmax + self.m + 1), 1, p = normalized)
        if dnew > _dmax:
            alpha = np.vstack((_alpha, alpha_stack[dnew]))
            beta  = np.vstack((_beta,  beta_stack[dnew]))
            delta = np.insert(_delta, i, dnew)
        else:
            delta = np.insert(_delta, i, dnew)
            alpha = _alpha
            beta  = _beta
        return delta, alpha, beta

    def sample_alpha_beta_new(self):
        alphas = self.alpha_prior.rvs(size = (self.m, self.d))
        betas  = np.hstack(ones(self.m), self.beta_prior.rvs(size = (self.m, self.d - 1)))
        return alphas, betas

    def update_alpha_beta(self, curr_alphas, deltas, Y):
        nClust = deltas.max() + 1
        djs  = [(delta == j) for j in range(nClust)]
        Yjks = [Y[djs[j], k] for j in range(nClust) for k in range(self.nCol)]
        idxs = list(range(self.nCol)) * nClust
        alpha_args = zip(idxs, curr_alphas.reshape(-1), Yjks, repeat(self.priors.alpha))
        prop_alphas = (pool.map(update_alpha_wrapper, alpha_args)).reshape(curr_alphas.shape)
        Yjks = [Y[djs[j], k] for k in range(1, self.nCol) for j in range(nClust)]
        beta_args = zip(prop_alphas[:,1:].reshape(-1), Yjks, repeat(self.priors.beta))
        prop_betas = np.hstack(
            np.ones(nClust),
            (pool.map(update_beta_wrapper), beta_args).reshape(nClust, self.nCol - 1),
            )
        return prop_alphas, prop_betas

    def sample_beta_j(self, alpha_j, Yj):
        """ Sampler for beta vector associated with cluster j """
        prop_beta = np.empty(self.nCol)
        prop_beta[0] = 1.
        for k in range(1,self.nCol):
            prop_beta[k] = sample_beta_fc(alpha_j[k], Yj[:,k], self.priors.beta)
        return prop_beta

    def sample_alpha_j(self, Yj):
        """ Sampler for the alpha vector associated with cluster j """
        # futures = []
        # futures.append(
        #     self.executor.submit(
        #         sample_alpha_1_slice,
        #         gmean(Yj[:,0]), Yj[:,0], self.priors.alpha
        #         )
        #     )
        # for k in range(1, self.nCol):
        #     futures.append(
        #         self.executor.submit(
        #             sample_alpha_k_slice,
        #             gmean(Yj[:,k]), Yj[:,k], self.priors.alpha, self.priors.beta,
        #             )
        #         )
        # prop_alpha = np.array([future.result() for future in futures])
        prop_alpha = np.empty(self.nCol)
        Yjg = gmean(Yj, axis = 0)

        prop_alpha[0] = sample_alpha_1_slice(
            Yjg[0], Yj[:,0], self.priors.alpha,
            )
        for k in range(1, self.nCol):
            prop_alpha[k] = sample_alpha_k_slice(
                Yjg[k], Yj[:,k], self.priors.alpha, self.priors.beta,
                )
        return prop_alpha

    def sample_theta_j(self, Yj):
        """ Sampler for theta parameter set associated with cluster j """
        alpha_j = self.sample_alpha_j(Yj)
        beta_j  = self.sample_beta_j(alpha_j, Yj)
        return Theta(alpha_j, beta_j)

    def sample_thetas(self, Y_ni, delta_ni):
        """ Samples a complete parameter set without the ith observation """
        delta_idx = [np.where(delta_ni == j) for j in range(delta_ni.max() + 1)]
        thetas    = [self.sample_theta_j(Y_ni[didx]) for didx in delta_idx]
        return thetas

    def sample_theta_new(self):
        alpha = self.alpha_prior.rvs(size = self.nCol)
        beta  = self.beta_prior.rvs(size = self.nCol - 1)
        beta  = np.insert(beta, 0, 1.)
        return Theta(alpha, beta)

    def sample_delta_i(self, R, delta, eta, i):
        # Compute latent Y's (without observation i)
        R_ni = np.delete(R, i)
        Yl_ni = np.delete(self.data.Yl, i, 0)
        Y_ni  = (Yl_ni.T * R_ni).T

        # Sample a new theta for each cluster (without obsv. i)
        delta_ni = self.clean_delta(np.delete(delta, i))
        theta_js = self.sample_thetas(Yl_ni, delta_ni)
        theta_js.append(self.sample_theta_new())

        # Compute log-likelihood under each cluster
        lls = np.array([
            logdprojgamma_pre(
                self.data.lsins[(i)], self.data.lcoss[(i)],
                self.data.Yl[(i)], theta.alpha, theta.beta,
                )
            for theta in theta_js
            ]).flatten()


        # gather number of observations per cluster
        dns = np.array(
            [(delta_ni == j).sum() for j in range(delta_ni.max() + 1)],
            dtype = np.float,
            )
        dns = np.insert(dns, dns.shape[0], eta)

        # Compute probability of cluster assignment
        unnormalized = np.exp(lls) * dns/dns.sum()
        normalized = unnormalized / unnormalized.sum()

        # Generate Cluster assignment, feed back into delta and return
        delta_i = choice(len(dns), 1, p = normalized)
        delta_new = np.insert(delta_ni, i, delta_i)
        return delta_new

    def sample_r(self, thetas, delta):
        alphas = np.array([thetas[didx].alpha for didx in delta])
        betas  = np.array([thetas[didx].beta for didx in delta])
        As = alphas.sum(axis = 1)
        Bs = (self.data.Yl * betas).sum(axis = 1)
        return gamma.rvs(As, scale = 1/Bs)

    def sample_eta(self, curr_eta, thetas):
        ns = len(thetas)
        g  = beta.rvs(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + ns
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma.rvs(aaa, bb)

    def set_priors(self):
        self.alpha_prior = gamma(self.priors.alpha.a, 1 / self.priors.alpha.b)
        self.beta_prior  = gamma(self.priors.beta.a,  1 / self.priors.beta.b)
        return

    def initialize_sampler(self, ns):
        self.samples_delta = np.empty((ns + 1, self.nDat), dtype = np.int)
        self.samples_r     = np.empty((ns + 1, self.nDat))
        self.samples_eta   = np.empty(ns + 1)
        self.samples_theta = []
        # Initial conditions
        self.samples_delta[0] = np.array(list(range(self.nDat)))
        self.samples_r[0]     = 1.
        self.samples_eta[0]   = 5.
        # self.samples_theta.append(
        #     [Theta(np.array([1.] * self.nCol), np.array([1.] * self.nCol))] * self.nDat
        #     )
        self.samples_theta.append(())
        return

    def sample(self, ns):
        self.initialize_sampler(ns + 1)
        for k in range(1, ns + 1):
            # Copy over previous delta set
            print('\rSampling {:.1%} Complete'.format(k/ns), end = '')
            self.samples_delta[k] = self.samples_delta[k-1]
            # Sample a new delta set
            for i in range(self.nDat):
                self.samples_delta[k] = self.sample_delta_i(
                        self.samples_r[k - 1],
                        self.samples_delta[k],
                        self.samples_eta[k - 1],
                        i,
                        )
            Y = (self.data.Yl.T * self.samples_r[k-1]).T
            thetas = self.sample_thetas(Y, self.samples_delta[k])
            self.samples_theta.append(thetas)
            self.samples_r[k]   = self.sample_r(
                    self.samples_theta[k],
                    self.samples_delta[k],
                    )
            self.samples_eta[k] = self.sample_eta(
                    self.samples_eta[k-1],
                    self.samples_delta[k],
                    )
        # Removing first sample
        self.samples_delta = self.samples_delta[1:]
        self.samples_eta   = self.samples_eta[1:]
        self.samples_theta = self.samples_theta[1:]
        self.samples_r     = self.samples_r[1:]
        return


    def __init__(
            self,
            data,
            prior_alpha = GammaPrior(1.,1.),
            prior_beta = GammaPrior(1.,1.),
            prior_eta = GammaPrior(4.,1.),
            ):
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = BNPPGPrior(
            prior_alpha,
            prior_beta,
            prior_eta,
            )
        self.set_priors()
        # self.executor = ThreadPoolExecutor()
        return

# EOF
