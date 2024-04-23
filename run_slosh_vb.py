from __future__ import division
import sys, os, glob, re
import numpy as np
import pandas as pd
import multiprocessing as mp
import sqlite3 as sql
from io import BytesIO
from time import sleep
from numpy.random import uniform
import pickle as pkl

from energy import limit_cpu, postpred_loss_full, energy_score_full_sc
from data import Data_From_Sphere, Data_From_Raw
import varbayes as vb

raw_path  = './datasets/slosh/filtered_data.csv.gz'
out_sql   = './datasets/slosh/results.sql'
out_table = 'energy'

def run_model_from_index(df, col_index, quantile = 0.95):
    raw = df[:,col_index]
    data = Data_From_Raw(raw, False, quantile = 0.95)
    model = vb.VarPYPG(data)
    model.fit_advi()
    pp = model.generate_posterior_predictive_hypercube()
    


def run_model_from_path(path, *pargs):
    basepath, fname = os.path.split(path)
    raw = pd.read_csv(path).values
    testpath = os.path.join(basepath, 'test' + fname[4:])
    if not os.path.exists(testpath):
        return
    test = pd.read_csv(testpath).values
    data = Data_From_Sphere(raw)
    
    model = vb.VarPYPG(data)
    model.fit_advi()
    pp = model.generate_posterior_predictive_hypercube(5000)
    
    es1 = energy_score_full_sc(pp, data.V)
    es2 = energy_score_full_sc(pp, test)
    esbl1 = energy_score_full_sc(data.V, test)
    esbl2 = energy_score_full_sc(test, data.V)
    
    df = pd.DataFrame([{
        'path'   : path,
        'model'  : 'VarPYPG',
        'es1'    : es1,
        'es2'    : es2,
        'esbl1'  : esbl1,
        'esbl2'  : esbl2,
        'time'   : model.time_elapsed
        }])
    conn = sql.connect(out_sql)
    for _ in range(10):
        try:
            df.to_sql(out_table, conn, if_exists = 'append', index = False)
            conn.commit()
            break
        except sql.OperationalError:
            sleep(uniform())
            pass
    
    conn.close()
    return

if __name__ == '__main__':
    slosh = pd.read_csv(
        './datasets/slosh/filtered_data.csv.gz', 
        compression = 'gzip',
        )
    slosh_ids = slosh.T[:8].T
    slosh_obs = slosh.T[8:].T
    sloshltd  = ~slosh.MTFCC.isin(['C3061','C3081'])

    sloshltd_ids = slosh_ids[sloshltd]
    sloshltd_obs = slosh_obs[sloshltd].values.T.astype(np.float64)

    data = Data_From_Raw(sloshltd_obs, decluster = False, quantile = 0.95)
    model = vb.VarPYPG(data)
    model.fit_advi()
    postalphas = model.generate_conditional_posterior_alphas()

    # write to disk
    d = {
        'ids'    : sloshltd_ids,
        'obs'    : sloshltd_obs,
        'alphas' : postalphas,
        }
    with open('./datasets/slosh/sloshltd.pkl', 'wb') as file:
        pkl.dump(d, file)

# EOF
