""" 
Module for implementing anomaly detection algorithms.

Implements classic anomaly detection algorithms, as well as custom anomaly detection algorithms for extreme data.
"""
# builtins imported as package
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import re, os, argparse, glob, gc
# builtins explicitly imported
from multiprocessing import pool as mcpool, cpu_count, Pool # get_context
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_func, gammaln
from scipy.stats import gmean
from numpy.random import gamma, choice
from itertools import repeat
from collections import defaultdict
from functools import cached_property
from time import sleep
from math import ceil
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from collections import defaultdict
# Competing Anomaly Detection Algorithms
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
# Custom Modules
from data import Projection, category_matrix, euclidean_to_catprob,             \
    euclidean_to_hypercube
from energy import limit_cpu, kde_per_obs, manhattan_distance_matrix,           \
    hypercube_distance_matrix, euclidean_distance_matrix,                       \
    euclidean_dmat_per_obs, hypercube_dmat_per_obs, manhattan_dmat_per_obs,     \
    mixed_energy_score, real_energy_score, simp_energy_score
from models import Results
np.seterr(divide = 'ignore')

# Globals
EPS = np.finfo(float).eps
MAX = np.finfo(float).max

# class DAMEX(object):
#     damex = defaultdict(lambda: 1e-6)
#     threshold = None
    
#     def __init__(self, data):
#         n = data.E.shape[0]
#         k = np.sqrt(data.E.shape[0])
#         epsilon = 0.01
#         self.threshold = n * epsilon / k
#         Z = np.array(list(map(std_pareto_transform, self.data.E.T))).T
#         U = (Z > self.threshold)
#         for u in map(tuple, U):
#             self.damex[u] += n / k

#     def score(self, X):
#         R = X.max(axis = 1)
#     pass

def metric_auc(scores, actual):
    """  
    metric_auc(scores, actual)
    ---
    Compute Area-Under-the-Curve statistics for ROC and PRC
    ---
    Args:
        scores : (N) vector, float
        actual : (N) vector, int/bool
    ---
    Out:  tuple: (AuROC, AuPRC)
    """
    scores[np.isinf(scores)] = MAX
    scores[scores > MAX] = MAX
    try:
        auroc = roc_auc_score(actual, scores)
    except ValueError:
        return (np.nan, np.nan)
    precision, recall, thresholds = precision_recall_curve(actual, scores)
    auprc = auc(recall, precision)
    return (auroc, auprc)

class Anomaly(Projection):
    """ 
    Anomaly:

    Implements a variety of classic and experimental anomaly detection metrics.

    Usage:
        - Create composite class with (Result[model], Anomaly).
        - Instantiate.
        - Anomaly metrics will prefer to use existing distance metrics before generating new ones.
    Note:
        - Can declare multiprocessing.Pool first, so that *_distance_matrix will use 
            an existing pool rather than making a new one every time.
    """
    postpred_per_samp = 1

    # Parallelism
    pool = None
    def pools_open(self):
        # self.pool = get_context('spawn').Pool(
        self.pool = Pool(
            processes = (3 * cpu_count()) // 4, 
            initializer = limit_cpu,
            )
        return
    def pools_closed(self):
        self.pool.close()
        self.pool.join()
        del self.pool
        return
    
    @property
    def zeta_sigma(self):
        zetas = np.array([
            zeta[delta] 
            for zeta, delta 
            in zip(self.samples.zeta, self.samples.delta)
            ])
        try:
            sigmas = np.array([
                sigma[delta]
                for sigma, delta
                in zip(self.samples.sigma, self.samples.delta) 
                ])
        except AttributeError:
            sigmas = np.ones(zetas.shape)
        return zetas, sigmas
    @property
    def r(self):
        zetas, sigmas = self.zeta_sigma
        self.set_projection()
        r_shape = zetas[:,:,:self.nCol].sum(axis = 2)
        r_rate  = (sigmas[:,:,:self.nCol] * self.data.Yp[None,:,:]).sum(axis = 2)
        return gamma(r_shape, scale = 1 / r_rate)
    @property
    def rho(self):
        zetas, sigmas = self.zeta_sigma
        try:
            rho_shapes = zetas[:,:,self.nCol:] + self.data.W[None,:,:]
            rho_rates  = sigmas[:,:,self.nCol:]
            return gamma(rho_shapes, rho_rates)
        except AttributeError:
            return np.zeros((self.nSamp, self.nDat, 0))
    
    def energy_score(self):
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            Vnew = euclidean_to_hypercube(
                self.generate_posterior_predictive_gammas(self.postpred_per_samp)[:,:self.nCol]
                )
            Wnew = self.generate_posterior_predictive_spheres(self.postpred_per_samp)
            return mixed_energy_score(self.data.V, self.data.W, Vnew, Wnew)
        elif hasattr(self.data, 'V'):
            Vnew = euclidean_to_hypercube(
                self.generate_posterior_predictive_gammas(self.postpred_per_samp)[:,:self.nCol]
                )
            return real_energy_score(self.data.V, Vnew)
        elif hasattr(self.data, 'W'):
            Wnew = self.generate_posterior_predictive_spheres(self.postpred_per_samp)
            return simp_energy_score(self.data.W, Wnew)
        else:
            raise            

    ## Latent Distance per Observation (Summary)
    def euclidean_distance(self, V = None, W = None, R = None, **kwargs):
        znew  = self.generate_new_conditional_posterior_predictive_zetas(Vnew = V, Wnew = W, Rnew = R)
        if hasattr(self.data, 'V') and (V is not None):            
            shape = znew[:,:,:self.nCol].sum(axis = 2)
            radii = gamma(shape).mean(axis = 1)
            Y1 = radii[:,None] * V
        elif W is not None:
            Y1 = np.zeros((W.shape[0], 0))
        else:
            raise ValueError('We need some data')
        if hasattr(self.data, 'W') and (W is not None):
            Y2 = gamma(znew[:,:,self.nCol:]).mean(axis = 1)
        elif V is not None:
            Y2 = np.zeros((V.shape[0], 0))
        else:
            raise ValueError('We need some data')
        Y = np.hstack((Y1,Y2))
        dmat = euclidean_distance_matrix(
            self.generate_posterior_predictive_gammas(self.postpred_per_samp),
            Y,
            self.pool,
            )
        return dmat
    def hypercube_distance(self, V = None, W = None, R = None):
        znew  = self.generate_new_conditional_posterior_predictive_zetas(Vnew = V, Wnew = W, Rnew = R)
        if hasattr(self.data, 'V') and (V is not None):            
            shape = znew[:,:,:self.nCol].sum(axis = 2)
            radii = gamma(shape).mean(axis = 1)
            Y1 = radii[:,None] * V
        elif W is not None:
            Y1 = np.zeros((W.shape[0], 0))
        else:
            raise ValueError('We need some data')
        if hasattr(self.data, 'W') and (W is not None):
            Y2 = gamma(znew[:,:,self.nCol:]).mean(axis = 1)
        elif V is not None:
            Y2 = np.zeros((V.shape[0], 0))
        else:
            raise ValueError('We need some data')
        Y = np.hstack((Y1,Y2))
        VV = euclidean_to_hypercube(Y)
        dmat = hypercube_distance_matrix(
            self.generate_posterior_predictive_gammas(self.postpred_per_samp),
            VV,
            self.pool,
            )
        return dmat
    def mixed_distance(self, V = None, W = None):
        Gcon = self.generate_new_conditional_posterior_predictive_gammas(Vnew = V, Wnew = W)
        Gnew = self.generate_posterior_predictive_gammas(self.postpred_per_samp)
        catmat = category_matrix(self.data.cats)
        if hasattr(self.data, 'V') and (V is not None):
            Vnew = euclidean_to_hypercube(Gnew[:,:self.nCol])
            dmat_r = hypercube_distance_matrix(Vnew, V, self.pool)
        else:
            dmat_r = np.zeros((W.shape[0], Gnew.shape[0]))
        if hasattr(self.data, 'W') and (W is not None):
            mrho = Gcon[:,:,self.nCol:].mean(axis = 1)
            pi_new = euclidean_to_catprob(Gnew, catmat)
            pi_con = euclidean_to_catprob(mrho, catmat)

            dmat_c = manhattan_distance_matrix(pi_new, pi_con, self.pool)
        else:
            dmat_c = np.zeros((V.shape[0], Gnew.shape[0]))
        return dmat_r, dmat_c

    # Bandwidth estimators
    @cached_property
    def hypercube_bandwidth(self):
        """hypercube bandwidth for only hypercube section"""
        V = euclidean_to_hypercube(self.generate_posterior_predictive_gammas(1)[:,:self.nCol])
        VV = hypercube_dmat_per_obs(V[None], V, self.pool)
        return np.sqrt((VV**2).sum() / (2 * V.shape[0] * (V.shape[0] - 1)))
    @cached_property
    def latent_euclidean_bandwidth(self):
        Y = self.generate_posterior_predictive_gammas(1)
        YY = euclidean_dmat_per_obs(Y[None], Y, self.pool)
        return np.sqrt((YY**2).sum() / (2 * Y.shape[0] * (Y.shape[0] - 1)))
    @cached_property
    def latent_sphere_bandwidth(self):
        P = self.generate_posterior_predictive_spheres(1)
        PP = manhattan_dmat_per_obs(P[None], P, self.pool)
        return np.sqrt((PP**2).sum() / (2 * P.shape[0] * (P.shape[0] - 1)))
    @cached_property
    def latent_hypercube_bandwidth(self):
        V = euclidean_to_hypercube(self.generate_posterior_predictive_gammas(1))
        VV = hypercube_dmat_per_obs(V[None], V, self.pool)
        return np.sqrt((VV**2).sum() / (2 * V.shape[0] * (V.shape[0] - 1)))
    @cached_property
    def latent_mixed_bandwidth(self):
        V = euclidean_to_hypercube(
            self.generate_posterior_predictive_gammas(1)[:,:self.nCol]
            )
        P = self.generate_posterior_predictive_spheres(1)
        
        VV = hypercube_dmat_per_obs(V[None], V, self.pool)
        PP = manhattan_dmat_per_obs(P[None], P, self.pool)
        
        hV = np.sqrt((VV**2).sum() / (2 * V.shape[0] * (V.shape[0] - 1)))
        hP = np.sqrt((PP**2).sum() / (2 * P.shape[0] * (P.shape[0] - 1)))
        return (hV, hP)

    ## Classic Anomaly Metrics:
    def isolation_forest(self, V = None, W = None, **kwargs):
        """ Implements IsolationForest Method. Scores are arranged so larger = more anomalous """
        # if hasattr(self.data, 'raw'):
        #     forest = IsolationForest().fit(self.data.raw)
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            dat = np.hstack((self.data.V, self.data.W))
            forest = IsolationForest().fit(dat)
            datnew = np.hstack((V,W))
            raw = forest.score_samples(datnew)
        elif hasattr(self.data, 'V'):
            dat = self.data.V
            forest = IsolationForest().fit(dat)
            raw = forest.score_samples(V)
        elif hasattr(self.data, 'W'):
            dat = self.data.W
            forest = IsolationForest().fit(dat)
            raw = forest.score_samples(W)
        else:
            raise
        return raw.max() - raw + 1
    def local_outlier_factor(self, V = None, W = None, k = 5, **kwargs):
        """ Implements Local Outlier Factor.  k specifies the number of neighbors to fit to. """
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            dat = np.hstack((self.data.V, self.data.W))
            lof = LocalOutlierFactor(n_neighbors = k, novelty = True).fit(dat)
            raw = lof.score_samples(np.hstack((V,W)))
        elif hasattr(self.data, 'V'):
            dat = self.data.V
            lof = LocalOutlierFactor(n_neighbors = k, novelty = True).fit(dat)
            raw = lof.score_samples(V)
        elif hasattr(self.data, 'W'):
            dat = self.data.W
            lof = LocalOutlierFactor(n_neighbors = k, novelty = True).fit(dat)
            raw = lof.score_samples(W)
        else:
            raise
        return raw.max() - raw + 1
    def one_class_svm(self, V = None, W = None, **kwargs):        
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            dat = np.hstack((self.data.V, self.data.W))
            svm = OneClassSVM(gamma = 'auto').fit(dat)
            raw = svm.score_samples(np.hstack((V,W)))
        elif hasattr(self.data, 'V'):
            dat = self.data.V
            svm = OneClassSVM(gamma = 'auto').fit(dat)
            raw = svm.score_samples(V)
        elif hasattr(self.data, 'W'):
            dat = self.data.W
            svm = OneClassSVM(gamma = 'auto').fit(dat)
            raw = svm.score_samples(W)
        else:
            raise
        return raw.max() - raw + 1

    ## Extreme Anomaly Metrics:
    def knn_hypercube_distance_to_postpred(self, V = None, W = None, k = 5, **kwargs):
        knn = np.array(list(map(np.sort, self.hypercube_distance(V = V, W = W))))[:, k, 0]
        try:
            n = V.shape[0]
        except AttributeError:
            n = W.shape[0]
        p = self.tCol
        log_scores = (
            + np.log(k/n)
            + gammaln((p-1)/2 + 1)
            - ((p-1)/2) * np.log(np.pi)
            - (p-1) * np.log(knn)
            )        
        with np.errstate(under = 'ignore', over='ignore'):
            scores = np.exp(log_scores)
        scores[np.isnan(scores)] = MAX
        return scores
    def knn_euclidean_distance_to_postpred(self, V = None, W = None, k = 5, **kwargs):
        knn = np.array(list(map(np.sort, self.euclidean_distance(V = V, W = W))))[:, k, 0]
        try:
            n = V.shape[0]
        except AttributeError:
            n = W.shape[0]
        p = self.tCol
        log_scores = (
            + np.log(k/n)
            + gammaln((p-1)/2 + 1)
            - ((p-1)/2) * np.log(np.pi)
            - (p-1) * np.log(knn)
            )        
        with np.errstate(under = 'ignore', over='ignore'):
            scores = np.exp(log_scores)
        scores[np.isnan(scores)] = MAX
        return scores
    def populate_cones(self, epsilon):
        postpred = euclidean_to_hypercube(
            self.generate_posterior_predictive_gammas(self.postpred_per_samp),
            )
        C_damex = (postpred > epsilon)
        cones = defaultdict(lambda: EPS)
        for row in C_damex:
            cones[tuple(row)] += 1 / postpred.shape[0]
        return cones
    def cone_density(self, V = None, W = None, R = None, epsilon = 0.5, **kwargs):
        if V is None:
            return np.array([np.nan] * W.shape[0])
        n = V.shape[0]
        cone_prob = self.populate_cones(epsilon)
        scores = np.empty(n)
        znew = self.generate_new_conditional_posterior_predictive_zetas(Vnew = V, Wnew = W, Rnew = R)
        rho_new = gamma(znew[:,:,self.nCol:]).mean(axis = 1)
        r_new = gamma(znew[:,:,:self.nCol].sum(axis = 2)).mean(axis = 1)
        Vnew = euclidean_to_hypercube(np.hstack((r_new[:,None] * V, rho_new)))
        for i in range(n):
            scores[i] = 1 / cone_prob[tuple(Vnew[i] > epsilon)]
        return scores
    def hypercube_kernel_density_estimate(self, V = None, W = None, kernel = 'gaussian', **kwargs):
        if V is None:
            return np.array([np.nan] * W.shape[0])
        h = self.latent_hypercube_bandwidth
        Z = self.hypercube_distance(V, W) / h
        if kernel == 'gaussian':
            return 1 / (np.exp(- 0.5 * (Z**2)).mean(axis = (1,2)) + EPS)
        elif kernel == 'laplace':
            return 1 / (np.exp(-np.abs(Z)).mean(axis = (1,2)) + EPS)
        else:
            raise ValueError('requested kernel not available')
        pass
    def euclidean_kernel_density_estimate(self, V = None, W = None, kernel = 'gaussian', **kwargs):
        h = self.latent_euclidean_bandwidth
        Z = self.euclidean_distance(V, W) / h
        if kernel == 'gaussian':
            return 1 / (np.exp(- 0.5 * Z**2).mean(axis = (1,2)) + EPS)
        elif kernel == 'laplace':
            return 1 / (np.exp(-np.abs(Z)).mean(axis = (1,2)) + EPS)
        else:
            raise ValueError('requested kernel not available')
        pass
    def latent_sphere_kernel_density_estimate(self, V = None, W = None, R = None, kernel = 'gaussian', **kwargs):
        # if V is not None:
        #     return np.array([np.nan] * W.shape[0])
        h = self.latent_sphere_bandwidth
        Zcon = self.generate_new_conditional_posterior_predictive_zetas(Vnew = V, Wnew = W, Rnew = R)
        Gcon = gamma(Zcon[:,:,self.nCol:] + W[:,None])
        catmat = category_matrix(self.data.Cats)
        Pcon = euclidean_to_catprob(Gcon, catmat)
        Pnew = self.generate_posterior_predictive_spheres(self.postpred_per_samp)
        inv_scores = kde_per_obs(Pcon, Pnew, h, 'manhattan', self.pool)
        return 1 / (inv_scores + EPS)
    def latent_euclidean_kernel_density_estimate(self, V = None, W = None, R = None, kernel = 'gaussian', **kwargs):
        h = self.latent_euclidean_bandwidth
        Zcon = self.generate_new_conditional_posterior_predictive_zetas(Vnew = V, Wnew = W, Rnew = R)
        if V is not None:
            Rcon = gamma(Zcon[:,:,:self.nCol].sum(axis = 2))
            Y1 = Rcon[:,:,None] * V[:,None,:]
        else:
            Y1 = np.zeros((*Zcon.shape[:-1], 0))
        Y2 = gamma(Zcon[:,:,self.nCol:])
        Ycon = np.concatenate((Y1,Y2), axis = 2)
        Ynew = self.generate_posterior_predictive_gammas(self.postpred_per_samp)
        inv_scores = kde_per_obs(Ycon, Ynew, h, 'euclidean', self.pool)
        return 1 / (inv_scores + EPS)
    def latent_hypercube_kernel_density_estimate(self, V = None, W = None, R = None, kernel = 'gaussian', **kwargs):
        if V is None:
            return np.array([np.nan] * W.shape[0])
        h = self.latent_euclidean_bandwidth
        Zcon = self.generate_new_conditional_posterior_predictive_zetas(Vnew = V, Wnew = W, Rnew = R)
        if V is not None:
            Rcon = gamma(Zcon[:,:,:self.nCol].sum(axis = 2))
            Y1 = Rcon[:,:,None] * V[:,None,:]
        else:
            Y1 = np.zero
        Y2 = gamma(Zcon[:,:,self.nCol:])
        Vcon = euclidean_to_hypercube(np.concatenate((Y1,Y2), axis = 2))
        Vnew = euclidean_to_hypercube(
            self.generate_posterior_predictive_gammas(self.postpred_per_samp),
            )
        inv_scores = kde_per_obs(Vcon, Vnew, h, 'hypercube', self.pool)
        return 1 / (inv_scores + EPS)
    def latent_mixed_kernel_density_estimate(self, V = None, W = None, kernel = 'gaussian', **kwargs):
        if V is None:
            return np.array([np.nan] * W.shape[0])
        h_real = self.hypercube_bandwidth
        Gnew = self.generate_posterior_predictive_gammas(self.postpred_per_samp)
        Vnew = euclidean_to_hypercube(Gnew[:,:self.nCol])
        S1 = kde_per_obs(V[None], Vnew, h_real, 'hypercube', self.pool)
        S2 = 1 / self.latent_sphere_kernel_density_estimate(V, W)
        return 1 / (S1 * S2  + EPS)
        
    # scoring metrics
    @property
    def scoring_metrics(self):
        metrics = {
            'iso'    : self.isolation_forest,
            'lof'    : self.local_outlier_factor,
            'svm'    : self.one_class_svm,
            'kedp'   : self.knn_euclidean_distance_to_postpred,
            'khdp'   : self.knn_hypercube_distance_to_postpred,
            'cone'   : self.cone_density,
            'ekde'   : self.euclidean_kernel_density_estimate,
            'hkde'   : self.hypercube_kernel_density_estimate,
            'lhkde'  : self.latent_hypercube_kernel_density_estimate,
            'lekde'  : self.latent_euclidean_kernel_density_estimate,
            'lskde'  : self.latent_sphere_kernel_density_estimate,
            'lmkde'  : self.latent_mixed_kernel_density_estimate,
            }
        return metrics
    def get_scores(self, V, W, R):
        metrics = self.scoring_metrics.keys()
        density_metrics = ['khdp','kedp','cone','hkde','ekde','lskde','lekde','lhkde','lmkde']
        out = pd.DataFrame()
        for metric in metrics:
            print('s' + '\b'*11 + metric.ljust(10), end = '')
            sleep(1)
            temp = self.scoring_metrics[metric](V = V, W = W, R = R).ravel()
            temp[np.isnan(temp)] = MAX
            temp[temp > MAX] = MAX
            out[metric] = np.log(temp)
            if type(R) is np.ndarray:
                if (metric in density_metrics) and (R is not None):
                    with np.errstate(over='ignore', invalid='ignore'):
                        out['c' + metric] = out[metric] + 2 * np.log(R)
        print('s' + '\b'*11 + 'Done'.ljust(10))
        return out
    def get_scoring_metrics(self, Y, V = None, W = None, R = None):
        scores = self.get_scores(V, W, R)
        aucs = np.array([metric_auc(score, Y) for score in scores.values.T]).T
        metrics = pd.DataFrame(aucs, columns = scores.columns.values.tolist())
        metrics['Metric'] = ('AuROC','AuPRC')
        metrics['EnergyScore'] = self.energy_score()
        return metrics
    def set_postpred_per_sample(self, n):
        self.postpred_per_samp = n
        return

def ResultFactory(model, path):
    class Result(Results[model], Anomaly):
        pass

    return Result(path)

def plot_log_inverse_scores(scores):
    plt.plot(np.sort(np.log(1/scores)))
    plt.show()
    return

def plot_log_inverse_scores_knn(scores):
    ord = np.argsort(scores.mean(axis = 1))
    plt.plot(np.log(1/scores[ord[::-1]]))
    plt.show()
    return

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('in_path')
    p.add_argument('out_path')
    return p.parse_args()

if __name__ == '__main__':
    pass

    result_path = './ad/{}/rank_results_1e-1_1e-1.pkl'
    datasets = ['annthyroid','cardio','cover','mammography','pima','yeast']
    result_paths = []
    metrics = []
    for dataset in datasets:
        result_files = glob.glob(result_path.format(dataset))
        for result_file in result_files:
            result_paths.append(result_file)
    
    pool = Pool(processes = ceil(0.8 * cpu_count()), initializer = limit_cpu)
    for result_path in result_paths:
        print(result_path.ljust(80), end = '')
        result = ResultFactory('pypprgln', result_path)
        result.pool = pool
        data = (result.data.Y, result.data.V, result.data.W, result.data.R)
        metric = result.get_scoring_metrics(*data)
        metric['path'] = result_path
        metrics.append(metric)
        del result
        gc.collect()
    
    pool.close()
    df = pd.concat(metrics)
    df.to_csv('./ad/performance_rank_radius.csv', index = False)
    

    # basepath = './ad/cardio'

    # result = ResultFactory('mpypprgln', os.path.join(basepath, 'results_xv1_1e-1_1e0.pkl'))
    # result.pools_open()

    # is_raw = pd.read_csv(os.path.join(basepath, 'data_xv1_is.csv')).values
    # os_raw = pd.read_csv(os.path.join(basepath, 'data_xv1_os.csv')).values
    # is_out  = pd.read_csv(os.path.join(basepath, 'outcome_xv1_is.csv')).values.ravel()
    # os_out  = pd.read_csv(os.path.join(basepath, 'outcome_xv1_os.csv')).values.ravel()
    # is_raw = is_raw[~np.isnan(is_raw).any(axis = 1)]
    # os_raw = os_raw[~np.isnan(os_raw).any(axis = 1)]
    # is_out  = is_out[~np.isnan(is_out)].astype(int)
    # os_out  = os_out[~np.isnan(os_out)].astype(int)

    # is_data = result.data.to_mixed_new(is_raw, is_out)
    # os_data = result.data.to_mixed_new(os_raw, os_out)
    
    # metric_is = result.get_scoring_metrics(*is_data)
    # metric_os = result.get_scoring_metrics(*os_data)

    # result.pools_closed()

    # raise
    # import re
    # results  = []
    # basepath = './ad'
    # # datasets = ['cardio','cover','mammography','pima','satellite','annthyroid','yeast']
    # # resbases = {'mdppprgln' : 'results_xv*.pkl'}
    # datasets = ['cardio','cover','mammography','annthyroid','yeast']
    # # resbases = {'mpypprgln' : 'results*.pkl'}
    # resbases = {'mpypprgln' : 'results_xv*.pkl'}
    # # datasets = ['solarflare']
    # # resbases = {'cdppprgln' : 'results_xv*.pkl'}
    # for model in resbases.keys():
    #     for dataset in datasets:
    #         files = glob.glob(os.path.join(basepath, dataset, resbases[model]))
    #         for file in files:
    #             results.append((model, file))
    #             # if not 'xv' in file:
    #             #     results.append((model, file))
    # metrics = []
    # pool = Pool(processes = ceil(0.8 * cpu_count()), initializer = limit_cpu)
    # for result in results:
    #     extant_result = ResultFactory(*result)
    #     extant_result.set_postpred_per_sample(20)
    #     extant_result.pool = pool

    #     # Normal Code
    #     # raw = pd.read_csv(
    #     #         os.path.join(os.path.split(result[1])[0], 'data.csv'),
    #     #         ).values
    #     # raw = raw[~np.isnan(raw).any(axis = 1)]
    #     # out = pd.read_csv(
    #     #         os.path.join(os.path.split(result[1])[0], 'outcome.csv'),
    #     #         ).values.ravel()
    #     # out = out[~np.isnan(out.astype(float))].astype(int)
    #     # data = extant_result.data.to_mixed_new(raw, out)
    #     # print('Processing Result {}'.format(result[1]).ljust(80), end = '')
    #     # extant_metric = extant_result.get_scoring_metrics(*data)
    #     # del extant_result
    #     # extant_metric['path'] = result[1]
    #     # gc.collect()

    #     # Cross-Validation Code
    #     cv = re.search('xv(\d+)', result[1]).group(1)
    #     is_raw = pd.read_csv(
    #         os.path.join(os.path.split(result[1])[0], 'data_xv{}_is.csv'.format(cv)),
    #         ).values
    #     is_raw = is_raw[~np.isnan(is_raw).any(axis = 1)]
    #     os_raw = pd.read_csv(
    #         os.path.join(os.path.split(result[1])[0], 'data_xv{}_os.csv'.format(cv)),
    #         ).values
    #     os_raw = os_raw[~np.isnan(os_raw).any(axis = 1)]
        
    #     is_out = pd.read_csv(
    #         os.path.join(os.path.split(result[1])[0], 'outcome_xv{}_is.csv'.format(cv)),
    #         ).values.ravel()
    #     is_out = is_out[~np.isnan(is_out.astype(float))].astype(int)
    #     os_out = pd.read_csv(
    #         os.path.join(os.path.split(result[1])[0], 'outcome_xv{}_os.csv'.format(cv)),
    #         ).values.ravel()
    #     os_out = os_out[~np.isnan(os_out.astype(float))].astype(int)

    #     is_data = extant_result.data.to_mixed_new(is_raw, is_out)
    #     os_data = extant_result.data.to_mixed_new(os_raw, os_out)

    #     print('Processing Result {} IS'.format(result[1]).ljust(80), end = '')
    #     extant_metric_is = extant_result.get_scoring_metrics(*is_data)
    #     print('Processing Result {} OOS'.format(result[1]).ljust(80), end = '')
    #     extant_metric_os = extant_result.get_scoring_metrics(*os_data)

    #     del extant_result
    #     extant_metric_is['path'] = result[1]
    #     extant_metric_os['path'] = result[1]
    #     extant_metric_is['InSamp'] = True
    #     extant_metric_os['InSamp'] = False
    #     metrics.append(extant_metric_is)
    #     metrics.append(extant_metric_os)
    #     gc.collect()
    
    # df = pd.concat(metrics)
    # df.to_csv('./ad/performance_py_xv.csv')
    # df.to_csv('./ad/performance_py.csv')

# EOF   
