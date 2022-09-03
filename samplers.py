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
# from numba import jit, njit, prange, int32, set_num_threads
import warnings

# set_num_threads(4)

EPS = np.finfo(float).eps
MAX = np.finfo(float).max


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

# # @jit
# def dp_sample_cluster(delta, log_likelihood, prob, eta):
#     N = delta.shape[0]; J = log_likelihood.shape[1]
#     curr_cluster_state = np.bincount(delta, minlength=J)
#     cand_cluster_state = (curr_cluster_state == 0) * 1
#     scratch = np.empty(J)
#     ncandcluster = sum(cand_cluster_state)
#     for n in range(N):
#         scratch[:] = 0
#         curr_cluster_state[delta[n]] -= 1 
#         scratch += curr_cluster_state
#         scratch += cand_cluster_state * (eta / (ncandcluster + 1e-9))
#         with np.errstate(divide = 'ignore'):
#             np.log(scratch, out = scratch)
#         scratch += log_likelihood[n]
#         scratch -= scratch.max()
#         np.cumsum(scratch, out = scratch)
#         delta[n] = np.searchsorted(scratch, prob[n] * scratch[-1])
#         curr_cluster_state[delta[n]]+= 1
#         if cand_cluster_state[delta[n]]:
#             ncandcluster -= 1
#             cand_cluster_state[delta[n]] = 0
#     return delta

# # @njit(parallel = True)
# def bincount2D_jit(arr, M):
#     T = arr.shape[0]
#     N = arr.shape[1]
#     out = np.zeros((T, M), dtype = int32)
#     for t in prange(T):
#         for n in range(N):
#             out[t,arr[t,n]] += 1
#     return out

# # @njit(fastmath = True, parallel = True)
# def cumsoftmax2d(arr):
#     T = arr.shape[0]
#     J = arr.shape[1]
#     scratch = np.empty(T)
#     for t in prange(T):
#         scratch[t] = np.max(arr[t])
#         for j in range(J):
#             arr[t,j] = math.exp(arr[t,j] - scratch[t])
#         arr[t] = np.nancumsum(arr[t])
#         scratch[t] = arr[t,J-1]
#         for j in range(J):
#             arr[t,j] /= scratch[t]
#     return

# # @njit
# def down_1(cluster_state, dvec):
#     for t, d in enumerate(dvec):
#         cluster_state[t,d] -= 1
#     return

# # @njit
# def up_1(cluster_state, dvec):
#     for t, d in enumerate(dvec):
#         cluster_state[t,d] += 1
#     return

# # @njit
# def null_cand(cluster_state, dvec):
#     for t, d in enumerate(dvec):
#         cluster_state[t,d] = False
#     return

# # @njit
# def sum2d(arr):
#     m = np.zeros(arr.shape[0])
#     for j in range(arr.shape[1]):
#         m += arr.T[j]
#     return m

# # @njit(parallel = True, fastmath = True)
# def pt_dp_logpost(arr, logl, curr_state, cand_state, eta):
#     T = arr.shape[0]
#     arr[:] = 0.
#     arr += curr_state
#     n_cand = sum2d(cand_state)
#     cand_weight = np.zeros(T)
#     for t in prange(T):
#         cand_weight[t] = eta[t] / n_cand[t]
#         arr[t] += cand_state[t] * cand_weight[t]
#         for j in range(arr.shape[1]):
#             arr[t,j] = math.log(arr[t,j])
#     arr += logl
#     return

# def pt_dp_sample_cluster_old(delta, log_likelihood, prob, eta):
#     """
#     Args:
#         delta          : (T x N)
#         log_likelihood : (N x T x J)
#         prob ([type])  : (N x T)
#         eta ([type])   : (T)
#     """
#     T = delta.shape[0]
#     N = delta.shape[1]
#     J = log_likelihood.shape[2]
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", message="The TBB threading layer requires TBB version")
#         curr_cluster_state = bincount2D_jit(delta, J)
#         cand_cluster_state = (curr_cluster_state == 0)
#         prob += np.expand_dims(np.arange(T),0)
#         scratch = np.empty(curr_cluster_state.shape)
#         for n in range(N):
#             down_1(curr_cluster_state, delta.T[n])
#             pt_dp_logpost(scratch, log_likelihood[n], 
#                         curr_cluster_state, cand_cluster_state, eta)
#             cumsoftmax2d(scratch)
#             scratch += np.expand_dims(np.arange(T), 1)
#             delta.T[n] = np.searchsorted(scratch.ravel(), prob[n]) % J
#             up_1(curr_cluster_state, delta.T[n])
#             null_cand(cand_cluster_state, delta.T[n])
#     return

def pt_dp_sample_cluster(delta, log_likelihood, prob, eta):
    """
    Args:
        delta          : (T x N)
        log_likelihood : (N x T x J)
        prob ([type])  : (N x T)
        eta ([type])   : (T)
    """
    T, N, J = delta.shape[0], delta.shape[1], log_likelihood.shape[2]
    curr_cluster_state = bincount2D_vectorized(delta, J)
    cand_cluster_state = (curr_cluster_state == 0)
    scratch = np.empty(curr_cluster_state.shape)
    temps = np.arange(T)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        for n in range(N):
            curr_cluster_state[temps, delta.T[n]] -= 1
            scratch[:] = curr_cluster_state
            scratch += cand_cluster_state * (eta / (cand_cluster_state.sum(axis = 1) + EPS))[:,None]
            np.log(scratch, out = scratch)
            scratch[np.isnan(scratch)] = -np.inf
            scratch += log_likelihood[n]
            scratch -= scratch.max(axis = 1)[:,None]
            np.exp(scratch, out = scratch)
            np.cumsum(scratch, axis = 1, out = scratch)
            scratch /= scratch[:,-1][:,None]
            delta.T[n] = (prob[n][:,None] > scratch).sum(axis = 1)
            curr_cluster_state[temps, delta.T[n]] += 1
            cand_cluster_state[temps, delta.T[n]] = False
    return


if __name__ == '__main__':
    pass



# EOF
