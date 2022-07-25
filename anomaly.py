""" 
Module for implementing anomaly detection algorithms.

Implements classic anomaly detection algorithms, as well as custom anomaly detection algorithms for extreme data.
"""
from inspect import Attribute
from xml.dom.minidom import Attr
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import re, os, argparse, glob, gc
# builtins explicitly called
from multiprocessing import pool as mcpool, cpu_count, get_context
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_func
from scipy.stats import gmean
from numpy.random import gamma, choice
from itertools import repeat
from collections import defaultdict
from functools import cached_property
# Competing Anomaly Detection Algorithms
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from data import euclidean_to_hypercube, Projection
# Custom Modules
from energy import euclidean_dmat, hypercube_dmat, limit_cpu, \
                     hypercube_distance_matrix, euclidean_distance_matrix
from models import Results
np.seterr(divide = 'ignore')

from classify import Classifier

def auc(scores, actual):
    c = Classifier(scores, actual)
    return (c.auroc, c.auprc)

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
    # Parallelism
    pool = None
    def pools_open(self):
        self.pool = get_context('spawn').Pool(processes = cpu_count(), initializer = limit_cpu)
        return
    def pools_closed(self):
        self.pool.close()
        self.pool.join()
        del self.pool
        return

    @cached_property
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

    @cached_property
    def r(self):
        zetas, sigmas = self.zeta_sigma
        self.set_projection()
        r_shape = zetas[:,:,:self.nCol].sum(axis = 2)
        r_rate  = (sigmas[:,:,:self.nCol] * self.data.Yp[None,:,:]).sum(axis = 2)
        return gamma(r_shape, scale = 1 / r_rate)
    @cached_property
    def rho(self):
        zetas, sigmas = self.zeta_sigma
        try:
            rho_shapes = zetas[:,:,self.nCol:] + self.data.W[None,:,:]
            rho_rates  = sigmas[:,:,self.nCol:]
            return gamma(rho_shapes, rho_rates)
        except AttributeError:
            return np.zeros((self.nSamp, self.nDat, 0))
    
    ## Distance Metrics
    @cached_property
    def euclidean_distance(self):
        mr = self.r.mean(axis = 0)
        mrho = self.rho.mean(axis = 0) 
        Y = np.hstack((mr[:,None] * self.data.Yp, mrho))
        return euclidean_distance_matrix(
            self.generate_posterior_predictive_gammas(), Y, self.pool,
            )
    @cached_property
    def hypercube_distance(self):
        mr = self.r.mean(axis = 0)
        mrho = self.rho.mean(axis = 0)
        Y = np.hstack((mr[:,None] * self.data.Yp, mrho))
        V = euclidean_to_hypercube(Y)
        return hypercube_distance_matrix(
            self.generate_posterior_predictive_gammas(), V, self.pool,
            )
    @cached_property
    def hypercube_distance_real(self):
        Vnew = euclidean_to_hypercube(
            self.generate_posterior_predictive_gammas()[:,:self.nCol],
            )
        return hypercube_distance_matrix(Vnew, self.data.V, self.pool)
    @cached_property
    def sphere_distance_latent(self):
        pi_con = np.swapaxes(self.generate_conditional_posterior_predictive_spheres(), 0, 1) # (n, s, d)
        pi_new = self.generate_posterior_predictive_spheres() # (s,d)
        # s1 = choice(np.arange(pi_new.shape[0]), size = pi_new.shape[0]//2, replace = False)
        # s2 = choice(np.arange(pi_new.shape[0]), size = pi_new.shape[0]//2, replace = False)
        # res = self.pool.map(euclidean_dmat, zip(repeat(pi_new[s1]), pi_con[:,s2]))
        s = np.random.choice(pi_new.shape[0], pi_new.shape[0]//2, False)
        res = self.pool.map(euclidean_dmat, zip(repeat(pi_new), pi_con[:,s]))
        return np.array(list(res))
    @cached_property
    def euclidean_distance_latent(self):
        R = self.generate_conditional_posterior_predictive_radii() # (s,n)
        Y1 = R[:,:,None] * self.data.V[None,:,:] # (s,n,d1),
        Y2 = self.generate_conditional_posterior_predictive_gammas()[:,:,self.nCol:] # (s,n,d2)
        Y_con = np.swapaxes(np.concatenate((Y1,Y2), axis = 2), 0, 1) # (n,s,d) 
        Y_new = self.generate_posterior_predictive_gammas()          # (s,d)
        # s1 = choice(np.arange(R.shape[0]), size = R.shape[0]//2, replace = False)
        # s2 = choice(np.arange(R.shape[0]), size = R.shape[0]//2, replace = False)
        # res = self.pool.map(euclidean_dmat, zip(repeat(Y_new[s1]), Y_con[:,s2]))
        s = np.random.choice(Y_new.shape[0], Y_new.shape[0]//2, False)
        res = self.pool.map(euclidean_dmat, zip(repeat(Y_new), Y_con[:,s]))
        return np.array(list(res))
    @cached_property
    def hypercube_distance_latent(self):
        R = self.generate_conditional_posterior_predictive_radii() # (s,n)
        Y1 = R[:,:,None] * self.data.V[None,:,:] # (s,n,d1),
        Y2 = self.generate_conditional_posterior_predictive_gammas()[:,:,self.nCol:] # (s,n,d2)
        Y_con = np.swapaxes(np.concatenate((Y1,Y2), axis = 2), 0, 1) # (n, s, d)
        V_con = np.array(list(map(euclidean_to_hypercube, Y_con)))
        V_new = euclidean_to_hypercube(self.generate_posterior_predictive_gammas())
        # s1 = choice(np.arange(R.shape[0]), size = R.shape[0]//2, replace = False)
        # s2 = choice(np.arange(R.shape[0]), size = R.shape[0]//2, replace = False)
        # res = self.pool.map(hypercube_dmat, zip(repeat(V_new[s1]), V_con[:,s2]))
        s = np.random.choice(V_new.shape[0], V_new.shape[0]//2, False)
        res = self.pool.map(hypercube_dmat, zip(repeat(V_new), V_con[:,s]))
        return np.array(list(res))
    
    ## Classic Anomaly Metrics:
    def isolation_forest(self):
        """ Implements IsolationForest Method. Scores are arranged so larger = more anomalous """
        try:
            forest = IsolationForest().fit(self.data.VW)
            raw = forest.score_samples(self.data.VW)
        except AttributeError:
            try:
                forest = IsolationForest().fit(self.data.V)
                raw = forest.score_samples(self.data.V)
            except AttributeError:
                try:
                    forest = IsolationForest().fit(self.data.W)
                    raw = forest.score_samples(self.data.W)
                except AttributeError:
                    print("Where's the data?")
                    raise
        return raw.max() - raw + 1
    def local_outlier_factor(self, k = 20):
        """ Implements Local Outlier Factor.  k specifies the number of neighbors to fit to. """
        try:
            lof = LocalOutlierFactor(n_neighbors = k).fit(self.data.VW)
        except AttributeError:
            try:
                lof = LocalOutlierFactor(n_neighbors = k).fit(self.data.V)
            except AttributeError:
                try:
                    lof = LocalOutlierFactor(n_neighbors = k).fit(self.data.W)
                except AttributeError:
                    print("Where's the data?")
                    raise
        raw = lof.negative_outlier_factor_.copy()
        return raw.max() - raw + 1
    def one_class_svm(self):
        try:
            svm = OneClassSVM(gamma = 'auto').fit(self.data.VW)
            raw = svm.score_samples(self.data.VW)
        except AttributeError:
            try:
                svm = OneClassSVM(gamma = 'auto').fit(self.data.V)
                raw = svm.score_samples(self.data.V)
            except AttributeError:
                try:
                    svm = OneClassSVM(gamma = 'auto').fit(self.data.W)
                    raw = svm.score_samples(self.data.W)
                except AttributeError:
                    print('Where\'s the data?')
                    raise 
        return raw.max() - raw + 1

    ## Extreme Anomaly Metrics:
    def average_euclidean_distance_to_postpred(self, **kwargs):
        # return self.euclidean_distance.mean(axis = 1)
        return self.euclidean_distance_latent.mean(axis = (1,2))
    def average_hypercube_distance_to_postpred(self, **kwargs):
        # return self.hypercube_distance.mean(axis = 1)
        return self.hypercube_distance_latent.mean(axis = (1,2))
    def average_sphere_distance_to_postpred(self, **kwargs):
        return self.sphere_distance_latent.mean(axis = 1)
    def knn_hypercube_distance_to_postpred(self, k = 10, **kwargs):
        knn = np.array(list(map(np.sort, self.hypercube_distance)))[:,k]
        try:
            n, p = self.data.VW.shape
        except AttributeError:
            try:
                n, p = self.data.V.shape
            except AttributeError:
                try:
                    n, p = self.data.W.shape
                except AttributeError:
                    print('Where\'s the data?')
                    raise
        inv_scores =  (k / n) / (np.pi**((p-1)/2)/gamma_func((p-1)/2 + 1) * knn**(p-1))
        return 1 / inv_scores
    def knn_euclidean_distance_to_postpred(self, k = 10, **kwargs):
        knn = np.array(list(map(np.sort, self.euclidean_distance)))[:,k]
        try:
            n, p = self.data.VW.shape
        except AttributeError:
            try:
                n, p = self.data.V.shape
            except AttributeError:
                try:
                    n, p = self.data.W.shape
                except AttributeError:
                    print('Where\'s the data?')
                    raise
        inv_scores =  (k / n) / (np.pi**((p-1)/2)/gamma_func((p-1)/2 + 1) * knn**(p-1))
        return 1 / inv_scores
    def populate_cones(self, epsilon):
        postpred = euclidean_to_hypercube(self.generate_posterior_predictive_gammas())
        C_damex = (postpred > epsilon)
        cones = defaultdict(lambda: 1e-10)
        for row in C_damex:
            cones[tuple(row)] += 1 / postpred.shape[0]
        return cones
    def cone_density(self, epsilon = 0.5, **kwargs):
        cone_prob = self.populate_cones(epsilon)
        scores = np.empty(self.data.nDat)
        try:
            Y = euclidean_to_hypercube(
                    np.hstack((
                        self.samples.r.mean(axis = 0)[:, None] * self.data.V, 
                        self.samples.rho.mean(axis = 0)
                        ))
                    )
        except AttributeError:
            Y = self.data.V
        for i in range(self.nDat):
            scores[i] = cone_prob[tuple(Y[i] > epsilon)]
        return scores
    def hypercube_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        # temporary code:
        h = gmean(self.hypercube_distance.ravel())
        if kernel == 'gaussian':
            return 1 / np.exp(-(self.hypercube_distance / h)**2).mean(axis = (1,2))
        elif kernel == 'laplace':
            return 1 / np.exp(-np.abs(self.hypercube_distance / h)).mean(axis = (1,2))
        else:
            raise ValueError('requested kernel not available')
        pass
    def euclidean_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        h = gmean(self.euclidean_distance.ravel())
        if kernel == 'gaussian':
            return 1 / np.exp(-(self.euclidean_distance / h)**2).mean(axis = (1,2))
        elif kernel == 'laplace':
            return 1 / np.exp(-np.abs(self.euclidean_distance / h)).mean(axis = (1,2))
        else:
            raise ValueError('requested kernel not available')
        pass
    def latent_simplex_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        """ computes mean kde for  """
        h = self.sphere_distance_latent.mean()
        if kernel == 'gaussian':
            return np.sqrt(2 * np.pi) * h / np.exp(
                - (self.sphere_distance_latent / h)**2).mean(axis = (1,2)
                )
        elif kernel == 'laplace':
            return 2 * h / np.exp(
                - np.abs(self.sphere_distance_latent / h)).mean(axis = (1,2)
                )
        else:
            raise ValueError('requested kernel not available')
        pass
    def latent_euclidean_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        h = gmean(self.euclidean_distance_latent.ravel())
        if kernel == 'gaussian':
            return np.sqrt(2 * np.pi) * h / np.exp(
                - (self.euclidean_distance_latent / h)**2).mean(axis = (1,2)
                )
        elif kernel == 'laplace':
            return 2 * h / np.exp(
                - np.abs(self.euclidean_distance_latent / h)).mean(axis = (1,2)
                )
        else:
            raise ValueError('requested kernel not available')
        pass
    def latent_hypercube_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        h = gmean(self.euclidean_distance_latent.ravel())
        if kernel == 'gaussian':
            return np.sqrt(2 * np.pi) * h / np.exp(
                - (self.euclidean_distance_latent / h)**2).mean(axis = (1,2)
                )
        elif kernel == 'laplace':
            return 2 * h / np.exp(
                -np.abs(self.euclidean_distance_latent / h)).mean(axis = (1,2)
                )
        else:
            raise ValueError('requested kernel not available')
        pass
    def mixed_latent_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        d_real = self.hypercube_distance_real
        h_real = gmean(d_real.ravel())
        d_simp = self.sphere_distance_latent
        h_simp = d_simp.mean()
        if kernel == 'gaussian':
            s1 = np.exp(-(d_real/h_real)**2).mean(axis = (1,2))
            s2 = np.exp(-(d_simp/h_simp)**2).mean(axis = (1,2))
            return 1 / (s1 * s2)
        elif kernel == 'laplace':
            s1 = np.exp(-np.abs(d_real/h_real)).mean(axis = (1,2))
            s1 = np.exp(-np.abs(d_simp/h_simp)).mean(axis = (1,2))
            return (1 / (s1 * s2))
        else:
            raise ValueError('requested kernel not available')
        pass
    
    def combined_knn_hypercube_distance_to_postpred(self, **kwargs):
        return self.knn_hypercube_distance_to_postpred(**kwargs) * self.data.R
    def combined_knn_euclidean_distance_to_postpred(self, **kwargs):
        return self.knn_euclidean_distance_to_postpred(**kwargs) * self.data.R
    def combined_cone_density(self, **kwargs):
        return self.cone_density(**kwargs) * self.data.R
    def combined_hypercube_kernel_density_estimate(self, **kwargs):
        return self.hypercube_kernel_density_estimate(**kwargs) * self.data.R
    def combined_euclidean_kernel_density_estimate(self, **kwargs):
        return self.euclidean_kernel_density_estimate(**kwargs) * self.data.R
    def combined_latent_simplex_kernel_density_estimate(self, **kwargs):
        return self.latent_simplex_kernel_density_estimate(**kwargs) * self.data.R
    def combined_latent_euclidean_kernel_density_estimate(self, **kwargs):
        return self.latent_euclidean_kernel_density_estimate(**kwargs) * self.data.R
    def combined_latent_hypercube_kernel_density_estimate(self, **kwargs):
        return self.latent_hypercube_kernel_density_estimate(**kwargs) * self.data.R
    def combined_mixed_latent_kernel_density_estimate(self, **kwargs):
        return self.mixed_latent_kernel_density_estimate(**kwargs) * self.data.R


    # scoring metrics
    @property
    def scoring_metrics(self):
        metrics = {
            'if'     : self.isolation_forest,
            'lof'    : self.local_outlier_factor,
            'svm'    : self.one_class_svm,
            'aedp'   : self.average_euclidean_distance_to_postpred,
            'ahdp'   : self.average_hypercube_distance_to_postpred,
            'kedp'   : self.knn_euclidean_distance_to_postpred,
            'khdp'   : self.knn_hypercube_distance_to_postpred,
            'cone'   : self.cone_density,
            'ekde'   : self.euclidean_kernel_density_estimate,
            'hkde'   : self.hypercube_kernel_density_estimate,
            'lhkde'  : self.latent_hypercube_kernel_density_estimate,
            'lekde'  : self.latent_euclidean_kernel_density_estimate,
            'lskde'  : self.latent_simplex_kernel_density_estimate,
            'mlkde'  : self.mixed_latent_kernel_density_estimate,
            }
        if hasattr(self.data, 'R'):
            metrics.update({
                'ckhdp'  : self.combined_knn_hypercube_distance_to_postpred,
                'ckedp'  : self.combined_knn_euclidean_distance_to_postpred,
                'ccone'  : self.combined_cone_density,
                'chkde'  : self.combined_hypercube_kernel_density_estimate,
                'cekde'  : self.combined_euclidean_kernel_density_estimate,
                'clskde' : self.combined_latent_simplex_kernel_density_estimate,
                'clekde' : self.combined_latent_euclidean_kernel_density_estimate,
                'clhkde' : self.combined_latent_euclidean_kernel_density_estimate,
                'cmlkde' : self.combined_mixed_latent_kernel_density_estimate,
                })
        return metrics
    def get_scores(self):
        metrics = self.scoring_metrics.keys()
        scores = np.array(
            list([self.scoring_metrics[metric]().ravel() for metric in metrics])
            )
        return scores 
    def get_scoring_metrics(self):
        scores = self.get_scores()
        aucs = np.array([auc(score, self.data.Y) for score in scores]).T
        metrics = pd.DataFrame(aucs, columns = self.scoring_metrics.keys())
        metrics['Metric'] = ('AuROC','AuPRC')
        return metrics

def ResultFactory(model, path):
    class Result(Results[model], Anomaly):
        pass

    return Result(path)

def MixedResultFactory(path):
    if 'mdppprgln' in path:
        class Result(Results['mdppprgln'], Anomaly):
            pass
        return Result(path)
    elif 'mdppprg' in path:
        class Result(Results['mdppprg'], Anomaly):
            pass
        return Result(path)
    else: 
        raise ValueError('Wrong!')
    pass

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
    results = []
    basepath = './ad'
    datasets = ['cardio','cover','mammography']
    resbases = {
        # 'mdppprg' : 'result_mdppprg_*.pkl',
        'mdppprgln' : 'results_mdppprgln_*.pkl',
        }
    for model in resbases.keys():
        for dataset in datasets:
            files = glob.glob(os.path.join(basepath, dataset, resbases[model]))
            for file in files:
                results.append((model, file))
    metrics = []
    for result in results:
        print('Processing Result {}'.format(result[1]))
        extant_result = ResultFactory(*result)
        extant_result.p = 10.
        extant_result.pools_open()
        extant_metric = extant_result.get_scoring_metrics()
        extant_result.pools_closed()
        del extant_result
        extant_metric['path'] = result[1]
        metrics.append(extant_metric)
        gc.collect()
    
    df = pd.concat(metrics)
    df.to_csv('./ad/performance.csv')

    # path = './simulated/lnad/results_mdppprgln.pkl'
    # extant_result = ResultFactory('mdppprgln', path)
    # extant_result.p = 10
    # extant_result.pools_open()
    # scores = extant_result.get_scores()
    # extant_result.pools_closed()
    # raise

    # args = argparser()
    # args = {'in_path' : './sim_mixed_ad/results_mdppprg*.pkl', 'out_path' : './sim_mixed_ad/metrics.csv'}
    # files = glob.glob(args['in_path'])
    # metrics = []    
    # for file in files:
    #     match = re.search('results_([a-zA-Z]+)_(\d+)_(\d+)_(\d+).pkl', file)
    #     model, nmix, nreal, ncat = match.group(1, 2, 3, 4)
    #     # temporary code
    #     Y = pd.read_csv(os.path.join(os.path.split(file)[0], 'class_m{}.csv'.format(nmix)))
    #     result = ResultFactory(model, file)
    #     result.data.Y = Y.values.ravel()
    #     result.pools_open()
    #     metric = result.get_scoring_metrics()
    #     result.pools_closed()
    #     metric['Model'] = model
    #     metric['nMix'] = nmix
    #     metric['nReal'] = nreal
    #     metric['nCat'] = ncat
    #     column_order = ['Model','nMix','nReal','nCat','Metric'] + list(result.scoring_metrics.keys())
    #     metrics.append(metric[column_order])
    
    # df = pd.concat(metrics)
    # df.to_csv(args['out_path'], index = False)
    # path = './ad/cardio/results_mdppprgln_2_1e1.pkl'
    # result = MixedResultFactory(path)
    # result.p = 10.
    # result.pools_open()
    # metrics = result.get_scoring_metrics()
    # result.pools_closed()
    # raise
    # pass

# EOF   