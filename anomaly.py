""" Module for implementing anomaly detection algorithms;
    building on postpred_loss.Prediction_Gammas.prediction algorithm """
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from multiprocessing import Pool, cpu_count
from scipy.stats import gmean
from itertools import repeat

from energy import limit_cpu, hypercube_distance_unsummed
from postpred_loss import PostPredLoss, Prediction_Gammas, Results
from argparser import argparser_ad as argparser



class AnomalyDetector(PostPredLoss):
    """ Implements anomaly detection algorithms; uses Linf """
    def pairwise_distance_to_replicates(self):
        predicted = self.Linf(self.prediction())
        pool = Pool(processes = cpu_count(), initializer = limit_cpu)
        res = pool.map(hypercube_distance_unsummed, zip(np.moveaxis(predicted, 0, 1), self.data.V))
        pool.close()
        return np.hstack(list(res)).T

    def pdr_plot(self):
        pdr = self.pairwise_distance_to_replicates()
        pdrms = np.sort(pdr.mean(axis = 1))
        plt.plot(pdrms)
        plt.show()
        return

    def pairwise_distance_to_postpred(self, n_per_sample = 10):
        postpred = self.generate_posterior_predictive_hypercube(n_per_sample)
        pool = Pool(processes = cpu_count(), initializer = limit_cpu)
        res = pool.map(hypercube_distance_unsummed, zip(repeat(postpred), self.data.V))
        pool.close()
        return np.hstack(list(res)).T

    def pdp_plot(self, n_per_sample = 10):
        pdp = self.pairwise_distance_to_postpred(n_per_sample)
        pdpm = gmean(pdp, axis = 1)
        pdpms = np.sort(pdpm)
        plt.plot(pdpms)
        plt.show()
        return

    def knn_distance(self, k = 10, n_per_sample = 10):
        postpred = self.generate_posterior_predictive_hypercube(n_per_sample)
        pool = Pool(processes = cpu_count(), initializer = limit_cpu)
        res = pool.map(hypercube_distance_unsummed, zip(repeat(postpred), self.data.V))
        pool.close()
        return np.vstack([sorted(r.reshape(-1)) for r in res]).T[:k].T

    def knn_distance_plot(self, k = 20, n_per_sample = 10):
        knn = self.knn_distance(k, n_per_sample)
        ord = np.argsort(gmean(knn, axis = 1))
        knn2 = knn[ord].T[np.array([4,9,14,19],dtype = int)].T
        plt.plot(knn2)
        plt.show()
        return

    pass



def ResultFactory(model, path):
    class Result(Results[model], AnomalyDetector, Prediction_Gammas[model]):
        pass
    return Result(path)

if __name__ == '__main__':
    args = argparser()
    model = os.path.split(os.path.split(args.path)[0])[1]
    # path = path to result
    # --> load result
    result = ResultFactory(model, args.path)
    pass

# EOF
