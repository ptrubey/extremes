from __future__ import division
import sys, os, glob, re
import numpy as np
import pandas as pd
import multiprocessing as mp
import sqlite3 as sql
from io import BytesIO
from data import euclidean_to_hypercube

from energy import postpred_loss_full, energy_score_full_sc,                    \
    energy_score, postpred_loss_single
from data import Data_From_Sphere
from models import RealChains as Chains, RealResults as Results

data_path = './simulated/test/data.csv'
test_path = './simulated/test/test.csv'
model     = 'sdppprg' # dirichlet
out_path  = './simulated/test/result.csv'

def run_model_from_path(datapath, testpath, modeltype, outpath):
    raw = pd.read_csv(datapath).values
    test = pd.read_csv(testpath).values
    data = Data_From_Sphere(raw)
    model = Chains[modeltype](
        data, 
        prior_eta = (2., 1.), 
        p = 1,
        max_clust_count = 100, # should be fine?
        )    
    model.sample(50000)
    out = BytesIO()
    model.write_to_disk(out, 40000, 20)
    res = Results[modeltype](out)
    pp = euclidean_to_hypercube(
        res.generate_posterior_predictive_gammas(10)
        )
    px = euclidean_to_hypercube(
        res.generate_conditional_posterior_predictive_gammas()
        )

    es1 = energy_score_full_sc(pp, data.V)
    ppl1 = postpred_loss_full(pp, data.V)
    es2 = energy_score_full_sc(pp, test)
    ppl2 = postpred_loss_full(pp, test)
    esi = energy_score(px, data.V)
    ppli = postpred_loss_single(np.swapaxes(px, 0, 1), data.V).mean()
    esbl = energy_score_full_sc(test, data.V)
    pplbl = postpred_loss_full(test, data.V)

    df = pd.DataFrame([{
        'path'   : datapath,
        'model'  : modeltype,
        'es1'    : es1,
        'ppl1'   : ppl1,
        'es2'    : es2,
        'ppl2'   : ppl2,
        'esi'    : esi,
        'ppli'   : ppli,
        'esbl'   : esbl,
        'pplbl'  : pplbl
        }])
    df.to_csv(outpath, index = False)
    return

if __name__ == '__main__':
    run_model_from_path(data_path, test_path, model, out_path)
    
# EOF
