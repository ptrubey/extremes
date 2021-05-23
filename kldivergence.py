# Compute Distance / Divergence between Distributions
import numpy as np
import pandas as pd
import os, glob
from collections import namedtuple
from argparser import argparser_ppl as argparser
from energy import knn_kl_divergence, limit_cpu
from multiprocessing import Pool

import models, models_mpi

Results = {**models.Results, **models_mpi.Results}

class KLD(object):
    def kl_divergence(self, k = 10, n_per_sample = 10):
        prediction = self.generate_posterior_predictive_hypercube(n_per_sample)
        return knn_kl_divergence(self.data.V, prediction, k)

    pass

def ResultFactory(model, path):
    class Result(Results[model], KLD):
        pass

    return Result(path)

KLDResult = namedtuple('KLDResult', 'type name kld')

def kl_generation(model):
    try:
        result = ResultFactory(*model)
        kldr = KLDResult(
            model[0],
            os.path.splitext(os.path.split(model[1])[1])[0],
            result.kl_divergence(),
            )
    except pd.io.sql.DatabaseError:
        kldr = KLDResult(
            model[0],
            'Null',
            np.zeros(10),
            )
    return kldr

if __name__ == '__main__':
    args = argparser()
    model_types = sorted(Results.keys())

    models = []
    for model_type in model_types:
        mm = glob.glob(os.path.join(args.path, model_type, 'results*.db'))
        for m in mm:
            models.append((model_type, m))

    pool = Pool(processes = os.cpu_count(), initializer = limit_cpu)
    kldrs = pool.map(kl_generation, models)
    pool.close()
    df = pd.DataFrame(
        [(x[0],x[1],*x[2]) for x in kldrs],
        columns = ['type','name'] + ['k_{}'.format(i) for i in range(len(kldrs[0][2]))]
        )
    df.to_csv(os.path.join(args.path, 'kl_divergence_curves.csv'), index = False)

# EOF
