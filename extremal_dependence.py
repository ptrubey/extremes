from postpred_loss import Result as resultdict
from cUtility import find_neighbors, cluster_size_summary
from models import Results
import os, glob
import numpy as np, pandas as pd
from collections import namedtuple


def chi_ij(Vs):
    return ((Vs / Vs.mean(axis = 0)).min(axis = 1)).mean()

class ResultAddendum(object):
    """ adds analysis methods to a Result class """
    def pairwise_chis(self, n_per_sample = 10):
        H = self.generate_posterior_predictive_hypercube(n_per_sample)
        colsets = [(i,j) for i in range(self.nCol) for j in range(i)]
        Hsep = [
            H.T[np.array(colset, dtype = np.int)].T
            for colset in colsets
            ]
        res = list(map(chi_ij, Hsep))
        return (colsets, res)

    def neighborhood(self):
        return find_neighbors(self.samples.delta)

    def cluster_summary(self):
        return cluster_size_summary(self.samples.delta)

    pass

class Conditional_Gamma(object):
    def sample_from_conditional(indices, given):
        pass

class Conditional_ResGamma(object):
    def sample_from_conditional(indices, given):
        pass

def ResultFactory(model_type, model_path):
    class ResultNew(Results[model_type], ResultAddendum):
        pass
    return ResultNew(model_path)

if __name__ == '__main__':
    base_path = './output'
    model_types = [
            'md',   'dpd',   'mgd',   'dpgd',   'mprg',   'dpprg',   'mpg',   'dppg',
            # 'mdln', 'dpdln', 'mgdln', 'dpgdln', 'mprgln', 'dpprgln', 'mpgln', 'dppgln',
            'mdln', 'dpdln', 'dpgdln', 'mprgln', 'dpprgln', 'mpgln', 'dppgln',
            # removed mgdln while it re-runs.
            'dppn',
            ]

    models = []
    for model_type in model_types:
        model_paths = glob.glob(os.path.join(base_path, model_type, 'results_*.db'))
        for model_path in model_paths:
            models.append((model_type, model_path))

    EDR = namedtuple('EDR', 'type name cols chis')
    edrs = []
    for model in models:
        print('Processing {}-{}'.format(*model))
        result = ResultFactory(*model)
        edrs.append(EDR(
            model[0],
            os.path.splitext(os.path.split(model[1])[1])[0],
            *result.pairwise_chis(),
            ))
    df = pd.DataFrame(
        [(edr.type, edr.name, *edr.chis) for edr in edrs],
        columns = ['type','name'] + ['chi_{}_{}'.format(*cols) for cols in edrs[0].cols]
        )
    df.to_csv('./output/pairwise_extremal_dependence_coefs.csv', index = False)
