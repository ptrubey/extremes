from __future__ import division
import sys, os, glob
import numpy as np
import pandas as pd
import multiprocessing as mp
import sqlite3 as sql
from time import sleep
from numpy.random import uniform

from projgamma.energy import limit_cpu, energy_score_full_sc
from projgamma.data import Data, euclidean_to_psphere
from projgamma.model_pypprg import Chain, Result

source_path = './simulated/sphere2/data_m*_r*_i*.csv'
out_sql     = './simulated/sphere2/result_241015.sql'
out_table   = 'energy'

def run_model_from_path_wrapper(args):
    return run_model_from_path(*args)

def run_model_from_path(path):
    basepath, fname = os.path.split(path)
    raw = euclidean_to_psphere(pd.read_csv(path).values)
    testpath = os.path.join(basepath, 'test' + fname[4:])
    if not os.path.exists(testpath):
        return
    test = pd.read_csv(testpath).values
    data = Data.from_raw(raw, sphr_cols = np.arange(raw.shape[0]))    
    model = Chain(data, max_clust_count = 200)
    try:
        model.sample(40000)
    except: # (AssertionError, FloatingPointError, ValueError):
        print('\nFailed: {}\n'.format(path))
        return 
    outd = model.to_dict()
    res = Result(outd)
    pp = res.generate_posterior_predictive_hypercube(10)
    
    es1 = energy_score_full_sc(pp, data.V)
    es2 = energy_score_full_sc(pp, test)
    esbl1 = energy_score_full_sc(data.V, test)
    esbl2 = energy_score_full_sc(test, data.V)
    
    raise

    df = pd.DataFrame([{
        'path'   : path,
        'model'  : 'pypprg',
        'es1'    : es1,
        'es2'    : es2,
        'esbl1'  : esbl1,
        'esbl2'  : esbl2,
        'time'   : model.time_elapsed_numeric
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
    files = glob.glob(source_path)

    conn = sql.connect(out_sql)
    args = [(file,) for file in files]
    try:
        df = pd.read_sql('select * from energy;', conn)[['path',]]
        done = list(map(tuple, df.drop_duplicates().values))
        todo = list(set(args).difference(set(done)))
    except pd.io.sql.DatabaseError:
        todo = args
    conn.close()
    todo_len = len(todo)

    for item in todo:
        run_model_from_path_wrapper(item)
    
    # pool = mp.Pool(
    #     processes = mp.cpu_count(), 
    #     initializer = limit_cpu, 
    #     maxtasksperchild = 1,
    #     )
    # for i, _ in enumerate(pool.imap_unordered(run_model_from_path_wrapper, todo), 1):
    #     sys.stderr.write('\rdone {0:.2%}'.format(i/todo_len))
    # pool.close()
    # pool.join()

    # raise

# EOF
