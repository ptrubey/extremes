""" Module for implementing anomaly detection algorithms;
    building on postpred_loss.Prediction_Gammas.prediction algorithm """
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from multiprocessing import Pool, cpu_count
from scipy.stats import gmean
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_func
from itertools import repeat
from collections import defaultdict

from energy import limit_cpu, hypercube_distance_unsummed, postpred_loss_single, energy_score_inner
from data import Data_From_Raw
from postpred_loss import PostPredLoss, Prediction_Gammas, Results
from raw_anomaly import Anomaly, roc_curve, prc_curve
from simulate_data import DataAD
# from argparser import argparser_ad as argparser

class AnomalyDetector(PostPredLoss):
    """ Implements anomaly detection algorithms; uses Linf """
    data = None

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

    def populate_cones(self, epsilon):
        C_damex = (self.postpred > epsilon).astype(int)
        cones = defaultdict(lambda: 1e-10)
        for row in C_damex:
            cones[tuple(row)] += 1 / self.postpred.shape[0]
        return cones

    def scoring_cones(self, epsilon = 0.5):
        cone_prob = self.populate_cones(epsilon)
        scores = np.empty(self.data.nDat)
        for i in range(self.nDat):
            scores[i] = cone_prob[tuple(self.data.V[i] > epsilon)] / self.data.R[i]
        return scores

    def scoring_cones_angular(self, epsilon = 0.5):
        cone_prob = self.populate_cones(epsilon)
        scores = np.empty(self.data.nDat)
        for i in range(self.nDat):
            scores[i] = cone_prob[tuple(self.data.V[i] > epsilon)]
        return scores

    def scoring_pdr(self, scalar = 1., base = np.e):
        """ Inverse of average density of posterior predictive distribution * probability of seeing
        observation as far out as that."""
        pdrm = self.pairwise_distance_to_replicates().mean(axis = 1)
        n, p = self.data.V.shape
        # inv_scores = (base ** (-scalar * pdrm).T / self.data.R).T
        inv_scores = 1 / (pdrm ** (p-1)) / self.data.R
        return 1 / inv_scores

    def scoring_pdr_angular(self, scalar = 1., base = np.e):
        pdrm = self.pairwise_distance_to_replicates().mean(axis = 1)
        n, p = self.data.V.shape
        # inv_scores = base ** (-scalar * pdrm)
        inv_scores = 1 / (pdrm ** (p-1))
        return 1 / inv_scores

    def scoring_pdp(self, scalar = 1., base = np.e, n_per_sample = 10):
        pdp = self.pairwise_distance_to_postpred(n_per_sample)
        pdpm = gmean(pdp, axis = 1)
        inv_scores = (base ** (-scalar * pdpm).T / self.data.R).T
        return 1 / inv_scores

    def scoring_pdp_angular(self, scalar = 1., base = np.e, n_per_sample = 10):
        pdp = self.pairwise_distance_to_postpred(n_per_sample)
        pdpm = gmean(pdp, axis = 1)
        inv_scores = base ** (-scalar * pdpm)
        return 1 / inv_scores

    def scoring_knn(self, scalar = 1., base = np.e, k = 5, n_per_sample = 10):
        knn = self.knn_distance(k, n_per_sample).T[-1]
        n, p = self.data.V.shape
        # inv_scores = (base**(- scalar * knn).T / self.data.R).T
        inv_scores =  (k / n) / (np.pi**((p-1)/2)/gamma_func((p-1)/2 + 1) * knn**(p-1)) / self.data.R
        return 1 / inv_scores

    def scoring_knn_angular(self, scalar = 1., base = np.e, k = 5, n_per_sample = 10):
        knn = self.knn_distance(k, n_per_sample).T[-1]
        n, p = self.data.V.shape
        inv_scores =  (k / n) / (np.pi**((p-1)/2)/gamma_func((p-1)/2 + 1) * knn**(p-1))
        return 1 / inv_scores

    def scoring_ppl(self):
        predicted = self.Linf(self.prediction())
        ppl = postpred_loss_single(predicted, self.data.V)
        return ppl * self.data.R

    def scoring_ppl_angular(self):
        predicted = self.Linf(self.prediction())
        ppl = postpred_loss_single(predicted, self.data.V)
        return ppl

    def scoring_es(self):
        predicted = self.Linf(self.prediction())
        es = energy_score_inner(np.moveaxis(predicted, 0, 1), self.data.V)
        return es * self.data.R

    def scoring_es_angular(self):
        predicted = self.Linf(self.prediction())
        es = energy_score_inner(np.moveaxis(predicted, 0, 1), self.data.V)
        return es

    def instantiate_data(self, path, quantile = 0.95, decluster = True):
        """ path: raw data path """
        raw = pd.read_csv(path)
        self.data = Data_From_Raw(raw, decluster = decluster, quantile = quantile)
        self.postpred = self.generate_posterior_predictive_hypercube(10)
        return

    def instantiate_data_ad(self, path):
        self.data = DataAD(path)
        self.postpred = self.generate_posterior_predictive_hypercube(10)
        return

    def get_scores(self, scalar = 1., base = np.e, epsilon = 0.5):
        pdr = self.scoring_pdr(scalar, base)
        pdra = self.scoring_pdr_angular(scalar, base)
        pdp = self.scoring_pdp(scalar, base)
        pdpa = self.scoring_pdp_angular(scalar, base)
        knn = self.scoring_knn(scalar, base)
        knna = self.scoring_knn_angular(scalar, base)
        cone = self.scoring_cones(epsilon)
        conea = self.scoring_cones_angular(epsilon)
        ppl = self.scoring_ppl()
        ppla = self.scoring_ppl_angular()
        es  = self.scoring_es()
        esa = self.scoring_es_angular()
        return np.array((pdr, pdra, pdp, pdpa, knn, knna, cone, conea, ppl, ppla, es, esa))

    def get_auroc(self, scores):
        res = roc_curve(scores, self.anomaly.y[self.data.I[0]])
        _res = pd.DataFrame(
            res, columns = ('tpr','fpr'),
            ).groupby('fpr').max().reset_index()[['tpr','fpr']].values
        return trapezoid(*_res.T)

    def get_auprc(self, scores):
        res = prc_curve(scores, self.anomaly.y[self.data.I[0]])
        _res = pd.DataFrame(
            res, columns = ('ppv','tpr'),
            ).groupby('tpr').max().reset_index()[['ppv','tpr']].values
        return trapezoid(*_res.T)

    def get_metrics(self, scalar = 1., base = np.e, epsilon = 0.5):
        scores = self.get_scores(scalar, base, epsilon)
        auroc = np.array(list(map(self.get_auroc, scores)))
        auprc = np.array(list(map(self.get_auprc, scores)))
        return np.array((auroc, auprc))

    pass

def ResultFactory(model, path):
    class Result(Results[model], AnomalyDetector, Prediction_Gammas[model]):
        anomaly = None

        def instantiate_raw_anomaly(self, path_x, path_y):
            self.anomaly = Anomaly(path_x, path_y)
            return

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

def make_result(model, result_path, path_x, path_y, quantile, decluster):
    result = ResultFactory(model, result_path)
    result.instantiate_raw_anomaly(path_x, path_y)
    result.instantiate_data(path_x, quantile, decluster)
    return result

def make_result_ad(model, result_path, path_x, path_y):
    result = ResultFactory(model, result_path)
    result.instantiate_raw_anomaly(path_x, path_y)
    result.instantiate_data_ad(path_x)
    return result

if __name__ == '__main__':
    # args = argparser()
    # model = os.path.split(os.path.split(args.model_path)[0])[1]
    # result = ResultFactory(model, args.model_path)
    # result.instantiate_data(args.data_path, decluster=True)
    #
    # scores_c = result.scoring_cones()
    # scores_r = result.scoring_pdr()
    # scores_p = result.scoring_pdp()
    # scores_k = result.scoring_knn()
    # models = ['dphprg','mhprg','dphprg','dphprg','dphprg']
    # model_paths = [
    #     './ad/cardio/dphprg/results_2_1e-1.db',
    #     './ad/cover/mhprg/results_50.db',
    #     './ad/mammography/dphprg/results_2_1e-1.db',
    #     './ad/pima/dphprg/results_2_1e-1.db',
    #     './ad/satellite/dphprg/results_2_1e-1.db',
    #     ]
    # paths_x = [
    #     './datasets/ad_cardio_x.csv',
    #     './datasets/ad_cover_x.csv',
    #     './datasets/ad_mammography_x.csv',
    #     './datasets/ad_pima_x.csv',
    #     './datasets/ad_satellite_x.csv',
    #     ]
    # paths_y = [
    #     './datasets/ad_cardio_y.csv',
    #     './datasets/ad_cover_y.csv',
    #     './datasets/ad_mammography_y.csv',
    #     './datasets/ad_pima_y.csv',
    #     './datasets/ad_satellite_y.csv',
    #     ]
    # quantiles = [0.95, 0.95, 0.95, 0.95, 0.97]
    # decluster = [False] * 5
    # results = [make_result(*x) for x in zip(models, model_paths, paths_x, paths_y, quantiles, decluster)]

    models = repeat('dphprg')
    model_paths = [
        './simulated_ad/m5_c5/dphprg/results_2_1e-1.db',
        './simulated_ad/m5_c10/dphprg/results_2_1e-1.db',
        './simulated_ad/m10_c5/dphprg/results_2_1e-1.db',
        './simulated_ad/m10_c10/dphprg/results_2_1e-1.db',
        ]
    paths_x = [
        './simulated_ad/ad_sim_m5_c5_x.csv',
        './simulated_ad/ad_sim_m5_c10_x.csv',
        './simulated_ad/ad_sim_m10_c5_x.csv',
        './simulated_ad/ad_sim_m10_c10_x.csv',
        ]
    paths_y = [
        './simulated_ad/ad_sim_m5_c5_y.csv',
        './simulated_ad/ad_sim_m5_c10_y.csv',
        './simulated_ad/ad_sim_m10_c5_y.csv',
        './simulated_ad/ad_sim_m10_c10_y.csv',
        ]
    results = [make_result_ad(*x) for x in zip(models, model_paths, paths_x, paths_y)]


# EOF
