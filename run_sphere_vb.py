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
import varbayes as vb

source_path = './simulated/sphere2/data_m*_r*_i*.csv'
out_sql     = './simulated/sphere2/result_240723.sql'
out_table   = 'energy'

def run_model_from_path_wrapper(args):
    return run_model_from_path(*args)

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
    limit_cpu()
    files = glob.glob(source_path)

    conn = sql.connect(out_sql)
    args = [(file, 'VarPYPG') for file in files]
    try:
        df = pd.read_sql('select * from energy;', conn)[['path','model']]
        done = list(map(tuple, df.drop_duplicates().values))
        todo = list(set(args).difference(set(done)))
    except pd.io.sql.DatabaseError:
        todo = args
    conn.close()
    todo_len = len(todo)
    for i, arg in enumerate(todo):
        sys.stderr.write('\rdone {0:.2%}'.format((i+1) / todo_len))
        run_model_from_path_wrapper(arg)
    print('\nDone!')
        

# EOF
