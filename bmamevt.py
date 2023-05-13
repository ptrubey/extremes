from rpy2.robjects.packages import STAP
from rpy2.rinterface import StrSexpVector, IntSexpVector
from rpy2.robjects.conversion import Converter
import numpy as np
import sys, os, glob
import pandas as pd
import multiprocessing as mp
import sqlite3 as sql
from itertools import repeat
from energy import limit_cpu, postpred_loss_full, energy_score_full_sc
import time
from numpy.random import uniform

source_path = './simulated/sphere2/data_m*_r*_i*.csv'
out_sql     = './simulated/sphere2/result_bmamevt2.sql'
out_table   = 'energy'

nSim = 20000
nBurn = 15000
nPer = 1

def postpred_pairwise_betas(path, nsim, nburn, nper):
    with open('./bmamevt.R', 'r') as f:
        string = f.read()
    env = STAP(string, 'env')
    rpath = StrSexpVector((path,))
    rnsim = IntSexpVector((nsim,))
    rnburn = IntSexpVector((nburn,))
    rnper = IntSexpVector((nper,))
    out = env.postpred_pairwise_betas(rpath, rnsim, rnburn, rnper)
    return np.array(out)

def run_model_from_path(path, nsim, nburn, nper):
    basepath, fname = os.path.split(path)
    raw = pd.read_csv(path).values
    testpath = os.path.join(basepath, 'test' + fname[4:])
    if not os.path.exists(testpath):
        return
    test = pd.read_csv(testpath).values
    pp = postpred_pairwise_betas(path, nsim, nburn, nper)

    es1    = energy_score_full_sc(pp, raw)
    ppl1   = postpred_loss_full(pp, raw)
    es2    = energy_score_full_sc(pp, test)
    ppl2   = postpred_loss_full(pp, test)
    esbl1  = energy_score_full_sc(raw, test)
    pplbl1 = postpred_loss_full(raw, test)
    esbl2  = energy_score_full_sc(test, raw)
    pplbl2 = postpred_loss_full(test, raw)
    
    df = pd.DataFrame([{
        'path'   : path,
        'model'  : 'pairwise_betas',
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
    for _ in range(10):
        try:
            df.to_sql(out_table, conn, if_exists = 'append', index = False)
            conn.commit()
            break
        except sql.OperationalError:
            time.sleep(uniform())
            pass
        
    
    conn.close()
    return

def run_model_from_path_wrapper(args):
    return run_model_from_path(*args)

if __name__ == '__main__':
    files = glob.glob(source_path)
    files = [file for file in files if 'r2_' not in file]
    with sql.connect(out_sql) as conn:
        try: 
            files_done = pd.read_sql('select path from energy;', conn).values.T[0]
            files_to_do = list(set(files).difference(set(files_done)))
        except:
            files_to_do = files
    
    
    pool = mp.Pool(
        processes = int(0.8 * mp.cpu_count()),
        initializer = limit_cpu,
        maxtasksperchild = 1,
        )
    args = list(zip(files_to_do, repeat(nSim), repeat(nBurn), repeat(nPer)))
    args_len = len(args)
    for i, _ in enumerate(pool.imap_unordered(run_model_from_path_wrapper, args), 1):
       sys.stderr.write('\rdone {0:.2%}'.format(i/args_len))
    print('')
    # run_model_from_path(*args[0])
    
# EOF
