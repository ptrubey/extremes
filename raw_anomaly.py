""" implements base anomaly detection algorithms """

import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from collections import namedtuple
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from itertools import repeat

def roc(cutoff, scores, y):
    """ Compute True Positive Rate, False Positive Rate for a given threshold """
    preds = (scores > cutoff).astype(int)
    tpr = sum(preds * y) / sum(y)
    fpr = sum(preds * (1 - y)) / sum(1 - y)
    return np.array((tpr,fpr))

def prc(cutoff, scores, y):
    """ Compute Precision, Recall for a given Threshold """
    preds = (scores > cutoff).astype(int)
    pre = sum(preds * y) / sum(preds)
    rec = sum(preds * y) / sum(y)
    return np.array((pre, rec))

class Anomaly(object):
    def isolation_forest(self):
        forest = IsolationForest().fit(self.X)
        raw = forest.score_samples(self.X)
        return raw.max() - raw + 1

    def local_outlier_factor(self, k = 20):
        lof = LocalOutlierFactor(n_neighbors = k).fit(self.X)
        raw = lof.negative_outlier_factor_.copy()
        return raw.max() - raw + 1

    def one_class_svm(self):
        svm = OneClassSVM(gamma = 'auto').fit(self.X)
        raw = svm.score_samples(self.X)
        return raw.max() - raw + 1

    def get_scores(self):
        svm = self.one_class_svm().reshape(-1,1)
        lof = self.local_outlier_factor().reshape(-1,1)
        forest = self.isolation_forest().reshape(-1,1)
        return np.hstack((svm, lof, forest))

    def get_auroc(self):
        scores = self.get_scores()
        res = np.array(list(map(self.generate_roc, scores.T))) # this should be nExp * 50 * 2..
        auc = np.array(list(map(lambda x: trapezoid(*x.T), res)))
        return auc

    def get_auprc(self):
        scores = self.get_scores()
        res = np.array(list(map(self.generate_prc, scores.T))) # this should be nExp * 50 * 2..
        auc = np.array(list(map(lambda x: trapezoid(*x.T), res)))
        return auc

    def generate_roc(self, scores, logrange = True):
        if logrange:
            lbub = np.quantile(np.log(scores), (0.25, 0.99))
            space = np.exp(np.linspace(*lbub))
        else:
            lbub = np.quantile(scores, (0.25, 0.98))
            space = np.linspace(*lbub)

        res = np.array(list(map(roc, space, repeat(scores), repeat(self.y))))
        out = np.vstack((np.array((1.,1.)).reshape(1,2), res, np.array((0.,0.)).reshape(1,2)))
        return np.flip(out, axis = 0)

    def generate_prc(self, scores, logrange = True):
        if logrange:
            lbub = np.quantile(np.log(scores), (0.25, 0.99))
            space = np.exp(np.linspace(*lbub))
        else:
            lbub = np.quantile(scores, (0.25, 0.98))
            space = np.linspace(*lbub)

        res = np.array(list(map(prc, space, repeat(scores), repeat(self.y))))
        out = np.vstack((np.array((0.,1.)).reshape(1,2), res, np.array((1.,0.)).reshape(1,2)))
        return np.flip(out, axis = 0)

    def __init__(self, path_x, path_y): #, path_results):
        self.X = pd.read_csv(path_x).values
        self.y = pd.read_csv(path_y).values.reshape(-1)
        # self.results =
        return

if __name__ == '__main__':
    Path = namedtuple('Path','x y')
    paths = [
        Path('./datasets/ad_cardio_x.csv', './datasets/ad_cardio_y.csv'),
        Path('./datasets/ad_cover_x.csv', './datasets/ad_cover_y.csv'),
        Path('./datasets/ad_mammography_x.csv', './datasets/ad_mammography_y.csv'),
        Path('./datasets/ad_pima_x.csv', './datasets/ad_pima_y.csv'),
        Path('./datasets/ad_satellite_x.csv', './datasets/ad_satellite_y.csv'),
        ]

# EOF
