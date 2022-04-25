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

class ClassificationMetric(object):
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
        res = np.array(list(map(roc_curve, scores.T, repeat(self.y))))
        auc = np.array(list(map(lambda x: trapezoid(*x.T), res)))
        return auc

    def get_auprc(self):
        scores = self.get_scores()
        res = np.array(list(map(prc_curve, scores.T, repeat(self.y))))
        auc = np.array(list(map(lambda x: trapezoid(*x.T), res)))
        return auc

    # def __init__(self, path_x, path_y): #, path_results):
    #     self.X = pd.read_csv(path_x).values
    #     self.y = pd.read_csv(path_y).values.reshape(-1)
    #     return
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

if __name__ == '__main__':
    pass
    # Path = namedtuple('Path','x y')
    # paths = [
    #     Path('./datasets/ad_cardio_x.csv', './datasets/ad_cardio_y.csv'),
    #     Path('./datasets/ad_cover_x.csv', './datasets/ad_cover_y.csv'),
    #     Path('./datasets/ad_mammography_x.csv', './datasets/ad_mammography_y.csv'),
    #     Path('./datasets/ad_pima_x.csv', './datasets/ad_pima_y.csv'),
    #     Path('./datasets/ad_satellite_x.csv', './datasets/ad_satellite_y.csv'),
    #     ]
    # auroc = np.array([Anomaly(*path).get_auroc() for path in paths])
    # auprc = np.array([Anomaly(*path).get_auprc() for path in paths])
    # print('AuROC')
    # print(auroc)
    # print('AuPRC')
    # print(auprc)
    # colnames = ['SVM','LOF','IF']
    # rownames = ['cardio','cover','mammography','pima','satellite']

    # auroc_df = pd.DataFrame(auroc, columns = colnames)
    # auroc_df['data'] = rownames
    # auprc_df = pd.DataFrame(auprc, columns = colnames)
    # auroc_df['data'] = rownames

    # auroc_df.to_csv('./ad/auroc.csv')
    # auprc_df.to_csv('./ad/auprc.csv')

# EOF
