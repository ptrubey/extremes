from argparser import argparser_ppl as argparser
import numpy as np
import pandas as pd
import sqlite3 as sql
import os
import data
import glob
from numpy.random import gamma, choice
from collections import namedtuple
from postpred_loss import ResultFactory, PostPredLoss, Prediction_Gamma_Alter
from simulate_data import Data, Result

PPLResult = namedtuple('PPLResult', 'Type Scenario PPL_Linf ES_Linf PPL_Linf_F ES_Linf_F')

def ppl_generation(model):
    result = ResultFactory(*model)
    scenario = os.path.split(os.path.split(os.path.split(model[1])[0])[0])[1]
    pplr = PPLResult(
            model[0],
            scenario,
            result.posterior_predictive_loss_Linf(),
            result.energy_score_Linf(),
            result.postpred_loss_Linf_full(),
            result.energy_score_Linf_full(),
            )
    return pplr

class Gen(Result, PostPredLoss, Prediction_Gamma_Alter):
    nSamp = 1000
    pass

def gen_ppl_generation(data_path):
    scenario = os.path.split(os.path.split(data_path)[0])[1]
    gen = Gen(data_path)
    pplr = PPLResult(
        'Generative',
        scenario,
        gen.posterior_predictive_loss_Linf(),
        gen.energy_score_Linf(),
        gen.postpred_loss_Linf_full(),
        gen.energy_score_Linf_full(),
        )
    return pplr

if __name__ == '__main__':
    args = argparser()
    paths = glob.glob(os.path.join(args.path,'results_*'))
    model_types = ['sdpppg', 'sdppprg','sdpppgln','sdppprgln']
    models = []
    gens   = []

    for path in paths:
        for model_type in model_types:
            mm = glob.glob(os.path.join(path, model_type, 'results*.db'))
            for m in mm:
                models.append((model_type, m))
        gg = glob.glob(os.path.join(path, 'data.db'))
        for g in gg:
            gens.append(g)

    pplrs = []
    for model in models:
        print('Processing Model {}'.format(model[0]), end = ' ')
        try:
            pplrs.append(ppl_generation(model))
            print('Passed')
        except pd.io.sql.DatabaseError:
            print('Failed')
            pass

    for gen in gens:
        print('Processing gen {}'.format(os.path.split(os.path.split(gen)[0])[1]), end = ' ')
        try:
            pplrs.append(gen_ppl_generation(gen))
            print('Passed')
        except pd.io.sql.DatabaseError:
            print('Failed')
            pass

    df = pd.DataFrame(pplrs, columns = ('Type', 'Scenario', 'PPL_Linf','ES_Linf','PPL_Linf_F','ES_Linf_F'))
    df.to_csv(os.path.join(args.path, 'post_pred_loss_results.csv'), index = False)

# EOF
