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
    raw = pd.read_csv(path).values
    data = Data_From_Sphere(raw)
    model = Chains[modeltype](
        data, 
        prior_eta = (2., 1.), 
        p = 10,
        max_clust_count = 150,
        )
    model.sample(1000)
    out = BytesIO()
    model.write_to_disk(out, 500, 10)
    res = Results[modeltype](out)
    pp = res.generate_posterior_predictive_hypercube(10)
    es = energy_score_full_sc(pp, data.V)
    ppl = postpred_loss_full(pp, data.V)
    df = pd.DataFrame([{
        'path' : path,
        'model' : modeltype,
        'es'  : es,
        'ppl' : ppl,
        }])
    with sql.connect(out_sql) as conn:
        df.to_sql(out_table, conn, if_exists = 'append', index = False)
    return

if __name__ == '__main__':
    files = glob.glob(source_path)
    search_string = r'data_m(\d+)_r(\d+).csv'

    pool = mp.Pool(processes = mp.cpu_count(), initializer = limit_cpu)
    args = [(file, model) for file in files for model in models]
    res = pool.map(run_model_from_path, args)
    
# EOF