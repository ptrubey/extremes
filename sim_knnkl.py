from argparser import argparser_ppl as argparser
import numpy as np
import pandas as pd
import sqlite3 as sql
import os
import data
import glob
from numpy.random import gamma, choice
from collections import namedtuple
import models, models_mpi
from simulate_data import Data as SimData, Result as SimResult
from kldivergence import KLD
from multiprocessing import Pool, cpu_count
from energy import limit_cpu

Results = {**models.Results, **models_mpi.Results}

def ResultFactory(model, result_path, data_path):
    class R(Results[model], KLD):
        pass
    result = R(result_path)
    result.data = SimData(data_path)
    return result

KLResult = namedtuple('KLResult', 'Type Scenario KLD')

def kl_generation(result_path):
    model = os.path.split(os.path.split(result_path)[0])[1]
    scenario = os.path.split(os.path.split(os.path.split(result_path)[0])[0])[1]
    data_path = os.path.join(os.path.split(os.path.split(result_path)[0])[0], 'data.db')
    result = ResultFactory(model, result_path, data_path)
    kldr = KLResult(model, scenario, result.kl_divergence())
    return kldr

def kl_wrapper(path):
    if os.path.split(path)[1] == 'data.db':
        return gen_kl_generation(path)
    else:
        return kl_generation(path)
    pass

class Gen(SimResult, KLD):
    nSamp = 1000
    pass

def gen_kl_generation(data_path):
    scenario = os.path.split(os.path.split(data_path)[0])[1]
    gen = Gen(data_path)
    kldr = KLResult('Generative', scenario, gen.kl_divergence())
    return kldr

if __name__ == '__main__':
    args = argparser()
    paths = glob.glob(os.path.join(args.path,'sim_*'))
    model_types = ['dphpg','dphprg','dphprgln','dppn','vhpg']
    models = []
    gens   = []

    for path in paths:
        for model_type in model_types:
            mm = glob.glob(os.path.join(path, model_type, 'results*.db'))
            for m in mm:
                models.append(m)

        gg = glob.glob(os.path.join(path, 'data.db'))
        for g in gg:
            gens.append(g)

    pool = Pool(processes = cpu_count(), initializer = limit_cpu)
    kldrs = list(pool.map(kl_wrapper, models + gens))
    pool.close()

    df = pd.DataFrame(
        [(x[0],x[1],*x[2]) for x in kldrs],
        columns = ['type','scenario'] + ['k_{}'.format(i) for i in range(len(kldrs[0][2]))],
        )
    df.to_csv(os.path.join(args.path, 'kl_divergence.csv'), index = False)

# EOF
