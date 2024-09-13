from __future__ import division
import sys, os, glob, re
import numpy as np
import pandas as pd
import multiprocessing as mp
import sqlite3 as sql
from io import BytesIO
from time import sleep
from numpy.random import uniform

from energy import limit_cpu, postpred_loss_full, energy_score_full_sc
from data import Data_From_Sphere
from model_spypprg_vb import Chain, Result

source_path = './simulated/sphere2/data_m*_r*_i*.csv'
out_sql     = './simulated/sphere2/result_240912.sql'
out_table   = 'energy'

def run_model_from_path_wrapper(args):
    return run_model_from_path(*args)

def run_model_from_path(path, modeltype):
    basepath, fname = os.path.split(path)
    raw = pd.read_csv(path).values
    testpath = os.path.join(basepath, 'test' + fname[4:])
    if not os.path.exists(testpath):
        return
    test = pd.read_csv(testpath).values
    data = Data_From_Sphere(raw)
    
    model = Chain(data, p = 10, gibbs_samples = 1000,)
    try:
        model.sample(5000)
    except: # (AssertionError, FloatingPointError, ValueError):
        print('\nFailed: {}\n'.format(path))
        return 
    out = BytesIO()
    model.write_to_disk(out)
    res = Result(out)
    pp = res.generate_posterior_predictive_hypercube(10)
    
    es1 = energy_score_full_sc(pp, data.V)
    es2 = energy_score_full_sc(pp, test)
    esbl1 = energy_score_full_sc(data.V, test)
    esbl2 = energy_score_full_sc(test, data.V)
    
    df = pd.DataFrame([{
        'path'   : path,
        'model'  : 'MVarPYPG',
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

def argparser():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs = '+')
    return p.parse_args()

if __name__ == '__main__':
    files = glob.glob(source_path)

    pool = mp.Pool(
        processes = mp.cpu_count(), 
        initializer = limit_cpu, 
        maxtasksperchild = 1,
        )
    
    conn = sql.connect(out_sql)
    args = [(file, 'MVarPYPG') for file in files]
    try:
        df = pd.read_sql('select * from energy;', conn)[['path','model']]
        done = list(map(tuple, df.drop_duplicates().values))
        todo = list(set(args).difference(set(done)))
    except pd.io.sql.DatabaseError:
        todo = args
    todo_len = len(todo)
    for i, _ in enumerate(pool.imap_unordered(run_model_from_path_wrapper, todo), 1):
        sys.stderr.write('\rdone {0:.2%}'.format(i/todo_len))
    
    pool.close()
    pool.join()

    # raise

# EOF
