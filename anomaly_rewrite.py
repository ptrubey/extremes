""" 
Module for implementing anomaly detection algorithms.

Implements classic anomaly detection algorithms, as well as custom anomaly detection algorithms for extreme data.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import re, os, argparse, glob
# builtins explicitly called
from multiprocessing import pool, Pool, cpu_count
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_func
from itertools import repeat
from collections import defaultdict
from functools import cached_property
# Competing Anomaly Detection Algorithms
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from data import euclidean_to_hypercube
# Custom Modules
from energy import limit_cpu, hypercube_distance_matrix, euclidean_distance_matrix
from models import Results

class ClassificationMetric(object):
    """ Wrapper for establishing the typical classification metrics """
    @property
    def tpr(self):
        return self.tp / (self.tp + self.fn + 1e-10)
    @property
    def fpr(self):
        return self.fp / (self.fp + self.tn + 1e-10)
    @property
    def ppv(self):
        return self.tp / (self.tp + self.fp + 1e-10)

    def roc(self):
        return np.array((self.tpr, self.fpr))

    def prc(self):
        return np.array((self.ppv, self.tpr))

    def __init__(self, prediction, actual):
        self.tp = (prediction * actual).sum()
        self.tn = ((1 - prediction) * (1 - actual)).sum()
        self.fp = (prediction * (1 - actual)).sum()
        self.fn = ((1 - prediction) * actual).sum()
        return

def roc(cutoff, scores, y):
    """ Compute True Positive Rate, False Positive Rate for a given threshold """
    preds = (scores >= cutoff).astype(int)
    return ClassificationMetric(preds, y).roc()
def roc_curve(scores, y, logrange = True):
    """ Compute ROC Curve by varying threshold """
    if logrange:
        lbub = np.quantile(np.log(scores), (0.01, 0.99))
        space = np.exp(np.linspace(*lbub))
    else:
        lbub = np.quantile(scores, (0.01, 0.99))
        space = np.linspace(*lbub)
    res = np.array(list(map(roc, space, repeat(scores), repeat(y))))
    out = np.vstack((np.array((1.,1.)).reshape(1,2), res, np.array((0.,0.)).reshape(1,2)))
    return out[np.argsort(out.T[1])]
def prc(cutoff, scores, y):
    """ Compute Precision, Recall for a given Threshold """
    preds = (scores >= cutoff).astype(int)
    return ClassificationMetric(preds, y).prc()
def prc_curve(scores, y, logrange = True):
    """ Compute PRC Curve for varying threshold """
    if logrange:
        lbub = np.quantile(np.log(scores), (0.01, 0.99))
        space = np.exp(np.linspace(*lbub))
    else:
        lbub = np.quantile(scores, (0.01, 0.99))
        space = np.linspace(*lbub)
    res = np.array(list(map(prc, space, repeat(scores), repeat(y))))
    out = np.vstack((np.array((0.,1.)).reshape(1,2), res, np.array((1.,0.)).reshape(1,2)))
    return out[np.argsort(out.T[1])]

class Anomaly(object):
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
    pool = None
    
    def pools_open(self):
        self.pool = Pool(processes = cpu_count(), initializer = limit_cpu)
        return
    
    def pools_closed(self):
        self.pool.close()
        self.pool.join()
        del self.pool
        return

    ## Distance Metrics
    @cached_property
    def euclidean_distance(self):
        Y1 = self.samples.r.mean(axis = 0)[:,None] * self.data.V
        try:
            Y2 = self.samples.rho.mean(axis = 0)
            Y = np.hstack((Y1, Y2))
            return euclidean_distance_matrix(self.generate_posterior_predictive_gammas(), Y, self.pool)
        except AttributeError:
            return euclidean_distance_matrix(self.generate_posterior_predictive_gammas(), Y1, self.pool)
    @cached_property
    def hypercube_distance(self):
        Y1 = self.samples.r.mean(axis = 0)[:, None] * self.data.V
        try: 
            Y2 = self.samples.rho.mean(axis = 0)
            Y = euclidean_to_hypercube(np.hstack((Y1,Y2)))
            return hypercube_distance_matrix(self.generate_posterior_predictive_gammas(), Y, self.pool)
        except AttributeError:
            Y = euclidean_to_hypercube(Y1)
            return hypercube_distance_matrix(self.generate_posterior_predictive_gammas(), Y, self.pool)

    ## Classic Anomaly Metrics:
    def isolation_forest(self):
        """ Implements IsolationForest Method. Scores are arranged so larger = more anomalous """
        forest = IsolationForest().fit(self.data.VW)
        raw = forest.score_samples(self.data.VW)
        return raw.max() - raw + 1
    def local_outlier_factor(self, k = 20):
        """ Implements Local Outlier Factor.  k specifies the number of neighbors to fit to. """
        lof = LocalOutlierFactor(n_neighbors = k).fit(self.data.VW)
        raw = lof.negative_outlier_factor_.copy()
        return raw.max() - raw + 1
    def one_class_svm(self):
        svm = OneClassSVM(gamma = 'auto').fit(self.data.VW)
        raw = svm.score_samples(self.data.VW)
        return raw.max() - raw + 1
    
    ## Extreme Anomaly Metrics:
    def average_euclidean_distance_to_postpred(self, **kwargs):
        return self.euclidean_distance.mean(axis = 1)
    def average_hypercube_distance_to_postpred(self, **kwargs):
        return self.hypercube_distance.mean(axis = 1)
    def knn_hypercube_distance_to_postpred(self, k = 10, **kwargs):
        knn = np.array(list(map(np.sort, self.hypercube_distance)))[:,k]
        n, p = self.data.VW.shape
        inv_scores =  (k / n) / (np.pi**((p-1)/2)/gamma_func((p-1)/2 + 1) * knn**(p-1))
        return 1 / inv_scores
    def knn_euclidean_distance_to_postpred(self, k = 10, **kwargs):
        knn = np.array(list(map(np.sort, self.euclidean_distance)))[:,k]
        n, p = self.data.VW.shape
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
    def hypercube_kernel_density_estimate(self, kernel = 'gaussian', h = 1, **kwargs):
        # temporary code:
        h = self.hypercube_distance.mean()
        # 
        if kernel == 'gaussian':
            return 1 / (np.exp(-(self.hypercube_distance / h)**2) / np.sqrt(2 * np.pi)).mean(axis = 1)
        elif kernel == 'laplace':
            return 1 / np.exp(-np.abs(self.hypercube_distance / h)).mean(axis = 1)
        else:
            raise ValueError('requested kernel not available')
        pass
    def euclidean_kernel_density_estimate(self, kernel = 'gaussian', h = 1, **kwargs):
        h = self.euclidean_distance.mean()
        if kernel == 'gaussian':
            return 1 / (np.exp(-(self.euclidean_distance / h)**2) / np.sqrt(2 * np.pi)).mean(axis = 1)
        elif kernel == 'laplace':
            return 1 / np.exp(-np.abs(self.euclidean_distance / h)).mean(axis = 1)
        else:
            raise ValueError('requested kernel not available')
        pass
    
    ## Classification Performance Metrics:
    def get_auroc(self, scores):
        """ 
        Get Area under the Receiver Operating Characteristics Curve for matrix of anomaly scores.
        scores matrix (n x j) is arranged as: [data (n), method (j)]
        """
        res = np.array(list(map(roc_curve, scores.T, repeat(self.data.Y))))
        auc = np.array(list(map(lambda x: trapezoid(*x.T), res)))
        return auc
    def get_auprc(self, scores):
        """ 
        Get Area under the Precision/Recall Curve for matrix of anomaly scores.
        scores matrix (n x j) is arranged as: [data (n), method (j)]
        """
        res = np.array(list(map(prc_curve, scores.T, repeat(self.data.Y))))
        auc = np.array(list(map(lambda x: trapezoid(*x.T), res)))
        return auc

    @property
    def scoring_metrics(self):
        return {
            'if'   : self.isolation_forest,
            'lof'  : self.local_outlier_factor,
            'svm'  : self.one_class_svm,
            'aedp' : self.average_euclidean_distance_to_postpred,
            'ahdp' : self.average_hypercube_distance_to_postpred,
            'kedp' : self.knn_euclidean_distance_to_postpred,
            'khdp' : self.knn_hypercube_distance_to_postpred,
            'cone' : self.cone_density,
            'ekde' : self.euclidean_kernel_density_estimate,
            'hkde' : self.hypercube_kernel_density_estimate,
            }
    def get_scores(self):
        metrics = self.scoring_metrics.keys()
        scores = np.array(list([self.scoring_metrics[metric]().ravel() for metric in metrics])).T
        return scores # pd.DataFrame(scores, columns = metrics.keys())
    def get_scoring_metrics(self):
        scores = self.get_scores()
        auroc = self.get_auroc(scores)
        auprc = self.get_auprc(scores)
        metrics = pd.DataFrame(np.vstack((auroc, auprc)), columns = self.scoring_metrics.keys())
        metrics['Metric'] = ('AuROC','AuPRC')
        return metrics

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
    # args = argparser()
    args = {'in_path' : './sim_mixed_ad/results_mdppprg*.pkl', 'out_path' : './sim_mixed_ad/metrics.csv'}
    files = glob.glob(args['in_path'])
    metrics = []    
    for file in files:
        match = re.search('results_([a-zA-Z]+)_(\d+)_(\d+)_(\d+).pkl', file)
        model, nmix, nreal, ncat = match.group(1, 2, 3, 4)
        # temporary code
        Y = pd.read_csv(os.path.join(os.path.split(file)[0], 'class_m{}.csv'.format(nmix)))
        result = ResultFactory(model, file)
        result.data.Y = Y.values.ravel()
        result.pools_open()
        metric = result.get_scoring_metrics()
        result.pools_closed()
        metric['Model'] = model
        metric['nMix'] = nmix
        metric['nReal'] = nreal
        metric['nCat'] = ncat
        column_order = ['Model','nMix','nReal','nCat','Metric'] + list(result.scoring_metrics.keys())
        metrics.append(metric[column_order])
    
    df = pd.concat(metrics)
    df.to_csv(args['out_path'], index = False)
    pass

# EOF