"""
samplers.py
--------------
Proto-classes for MCMC samplers.

both assume the existence of:
    obj.initialize_sampler(ns)
    obj.iter_sample()
    obj.curr_iter

DirichletProcessSampler assumes the existence of:
    obj.samples.delta
    obj.curr_delta
"""
import time
import numpy as np
import math
from numba import jit, njit, prange


def bincount2D_vectorized(arr, m):
    """
    code from stackoverflow:
        https://stackoverflow.com/questions/46256279    
    Args:
        arr : (np.ndarray(int)) -- matrix of cluster assignments by temperature (t x n)
        m   : (int)             -- maximum number of clusters
    Returns:
        (np.ndarray(int)): matrix of cluster counts by temperature (t x J)
    """
    arr_offs = arr + np.arange(arr.shape[0])[:,None] * m
    return np.bincount(arr_offs.ravel(), minlength=arr.shape[0] * m).reshape(-1, m)

class BaseSampler(object):
    print_string_during = '\rSampling {:.1%} Completed in {}'
    print_string_after = '\rSampling 100% Completed in {}'

    @property
    def time_elapsed(self):
        """ returns current time elapsed since sampling start in human readable format """
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return '{:.0f} Seconds'.format(elapsed)
        elif elapsed < 3600: 
            return '{:.2f} Minutes'.format(elapsed / 60)
        else:
            return '{:.2f} Hours'.format(elapsed / 3600)
        pass

    def sample(self, ns):
        """ Run the Sampler """
        self.initialize_sampler(ns)
        self.start_time = time.time()
        
        print('\rSampling 0% Completed', end = '')

        while (self.curr_iter < ns):
            if (self.curr_iter % 100) == 0:
                ps = self.print_string_during.format(self.curr_iter / ns, self.time_elapsed)
                print(ps.ljust(80), end = '')
            self.iter_sample()
        
        ps = self.print_string_after.format(self.time_elapsed)
        print(ps)
        return

class DirichletProcessSampler(BaseSampler):
    print_string_during = '\rSampling {:.1%} Completed in {}, {} Clusters'
    print_string_after  = '\rSampling 100% Completed in {}, {} Clusters Avg.'
    
    @property
    def curr_cluster_count(self):
        """ Returns current cluster count """
        return self.curr_delta.max() + 1
    
    def average_cluster_count(self, ns):
        acc = self.samples.delta[(ns//2):].max(axis = 1).mean() + 1
        return '{:.2f}'.format(acc)

    def sample(self, ns):
        """ Run the sampler """
        self.initialize_sampler(ns)
        self.start_time = time.time()
        
        print('\rSampling 0% Completed', end = '')
        
        while (self.curr_iter < ns):
            if (self.curr_iter % 100) == 0:
                ps = self.print_string_during.format(
                    self.curr_iter / ns, self.time_elapsed, self.curr_cluster_count,
                    )
                print(ps.ljust(80), end = '')
            self.iter_sample()
        
        ps = self.print_string_after.format(self.time_elapsed, self.average_cluster_count(ns))
        print(ps)
        return

@jit
def dp_sample_cluster(delta, log_likelihood, prob, eta):
    N = delta.shape[0]; J = log_likelihood.shape[1]
    curr_cluster_state = np.bincount(delta, minlength=J)
    cand_cluster_state = (curr_cluster_state == 0) * 1
    scratch = np.empty(J)
    ncandcluster = sum(cand_cluster_state)
    for n in range(N):
        scratch[:] = 0
        curr_cluster_state[delta[n]] -= 1 
        scratch += curr_cluster_state
        scratch += cand_cluster_state * (eta / (ncandcluster + 1e-9))
        with np.errstate(divide = 'ignore'):
            np.log(scratch, out = scratch)
        scratch += log_likelihood[n]
        scratch -= scratch.max()
        np.cumsum(scratch, out = scratch)
        delta[n] = np.searchsorted(scratch, prob[n] * scratch[-1])
        curr_cluster_state[delta[n]]+= 1
        if cand_cluster_state[delta[n]]:
            ncandcluster -= 1
            cand_cluster_state[delta[n]] = 0
    return delta

@njit(parallel = True)
def bincount2D_jit(arr, M):
    T = arr.shape[0]
    N = arr.shape[1]
    out = np.zeros((T, M), dtype = int)
    for t in prange(T):
        for n in range(N):
            out[t,arr[t,n]] += 1
    return out

@njit(fastmath = True)
def fff(s, n):
    for i in range(1, n+1):
        s += math.exp(i * 0.0001)
    return s

@njit(fastmath = True, parallel = True)
def cumsoftmax3d(arr):
    N = arr.shape[0]
    T = arr.shape[1]
    J = arr.shape[2]
    scratch = np.empty(N,J)
    for n in prange(N):
        for t in prange(T):
            scratch[n,t] = np.max(arr[n,t])
            for j in range(J):
                arr[n,t,j] = math.exp(arr[n,t,j] - scratch[n,t,j])
            arr[n,t] = np.nancumsum(arr[n,t])
            np.divide(arr[n,t], arr[n,t,]) # no bueno



@njit(parallel = True)
def pt_dp_sample_cluster_jit(delta, log_likelihood, prob, eta):
    T = delta.shape[0]
    N = delta.shape[1]
    J = log_likelihood.shape[2]
    curr_cluster_state = bincount2D_jit(delta, J)
    cand_cluster_state = (curr_cluster_state == 0)
    for t in prange(T):
        pass

def pt_dp_sample_cluster(delta, log_likelihood, prob, eta):
    """
    Args:
        delta          : (T x N)
        log_likelihood : (T x N x J)
        prob ([type])  : (T x N)
        eta ([type])   : (T)
    """
    T = delta.shape[0]
    N = delta.shape[1]
    J = log_likelihood.shape[2]
    curr_cluster_state = bincount2D_jit(delta, J)
    cand_cluster_state = (curr_cluster_state == 0)
    tidx = np.arange(T)
    prob += np.expand_dims(tidx,0)
    scratch = np.empty(curr_cluster_state.shape)
    clust = np.empty_like(delta.T[0])
    for n in range(N):
        clust[:] = delta.T[n]
        curr_cluster_state[tidx, clust] -= 1
        scratch[:] = 0
        scratch += curr_cluster_state
        scratch += cand_cluster_state * np.expand_dims(
            eta / (np.sum(cand_cluster_state, axis = 1) + 1e-9), -1,
            )
        np.log(scratch, out = scratch)
        scratch += log_likelihood[n]
        scratch -= scratch.max()
        np.nan_to_num(scratch, False, -np.inf)
        np.exp(scratch, out = scratch)
        np.cumsum(scratch, axis = 1, out = scratch)
        scratch /= scratch.T[-1][:,None]
        scratch += tidx[:,None]
        delta.T[n] = np.searchsorted(scratch.ravel(), prob[n]) % J
        clust[:] = delta.T[n]
        curr_cluster_state[tidx, clust] += 1
        cand_cluster_state[tidx, clust] = False
    return delta




        

# EOF
