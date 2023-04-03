from __future__ import division
import sys, os, glob, re
import numpy as np
import pandas as pd
import multiprocessing as mp
import sqlite3 as sql
from io import BytesIO

from energy import limit_cpu, postpred_loss_full, energy_score_full_sc
from data import Data_From_Sphere
from models import RealChains as Chains, RealResults as Results

source_path = './simulated/sphere/data_m*_r*_i*.csv'
dest_path   = './simulated/sphere'
models      = ['sdpppg', 'sdppprg', 'sdpppgln', 'sdppprgln']
out_sql     = './simulated/sphere/result.sql'
out_table   = 'energy'

def run_model_from_path_wrapper(args):
    return run_model_from_path(*args)

def run_model_from_path(path, modeltype):
    basepath, fname = os.path.split(path)
    raw = pd.read_csv(path).values
    testpath = os.path.join(basepath, 'test' + fname[4:])
    if not os.path.exists(testpath):
        return
    test = pd.dread_csv(testpath)
    data = Data_From_Sphere(raw)
    
    model = Chains[modeltype](
        data, 
        prior_eta = (2., 1.), 
        p = 10,
        max_clust_count = 150,
        )
    
    model.sample(50000)
    out = BytesIO()
    model.write_to_disk(out, 40000, 20)
    res = Results[modeltype](out)
    pp = res.generate_posterior_predictive_hypercube(10)
    
    es1 = energy_score_full_sc(pp, data.V)
    ppl1 = postpred_loss_full(pp, data.V)
    es2 = energy_score_full_sc(pp, test)
    ppl2 = postpred_loss_full(pp, test)
    esbl1 = energy_score_full_sc(data.V, test)
    pplbl1 = postpred_loss_full(data.V, test)
    esbl2 = energy_score_full_sc(test, data.V)
    pplbl2 = postpred_loss_full(test, data.V)
    
    df = pd.DataFrame([{
        'path'   : path,
        'model'  : modeltype,
        'es1'    : es1,
        'ppl1'   : ppl1,
        'es2'    : es2,
        'ppl2'   : ppl2,
        'esbl1'  : esbl1,
        'pplbl1' : pplbl1,
        'esbl2'  : esbl2,
        'pplbl2' : pplbl2,
        }])
    conn = sql.connect(out_sql)
    df.to_sql(out_table, conn, if_exists = 'append', index = False)
    conn.commit()
    conn.close()
    return

def argparser():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs = '+')
    return p.parse_args()

if __name__ == '__main__':
    files = glob.glob(source_path)
    # search_string = r'data_m(\d+)_r(\d+)_i(\d+).csv'
    p = argparser()
    
    pool = mp.Pool(
        processes = mp.cpu_count(), 
        initializer = limit_cpu, 
        maxtasksperchild = 1,
        )
    
    args = [(file, model) for file in files for model in p.models]
    # args = [(file, p.model) for file in files]
    args_len = len(args)
    for i, _ in enumerate(pool.imap_unordered(run_model_from_path_wrapper, args), 1):
        sys.stderr.write('\rdone {0:.2%}'.format(i/args_len))
    res = pool.map(run_model_from_path_wrapper, args)
    pool.close()
    pool.join()

    # path = './simulated/sphere/data_m5_r3_i8.csv'
    # model = 'sdppprg'
    # run_model_from_path(path, model)

    # raise

# EOF
