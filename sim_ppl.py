from argparser import argparser_ppl as argparser
import numpy as np
import pandas as pd
import sqlite3 as sql
import os
import data
import glob
from numpy.random import gamma, choice
from collections import namedtuple
from postpred_loss import ResultFactory

PPLResult = namedtuple('PPLResult', 'Type Scenario PPL_Linf ES_Linf')

def ppl_generation(model):
    result = ResultFactory(*model)
    scenario = os.path.split(os.path.split(os.path.split(model[1])[0])[0])[1]
    pplr = PPLResult(
            model[0],
            scenario,
            result.posterior_predictive_loss_Linf(),
            result.energy_score_Linf(),
            )
    return pplr

if __name__ == '__main__':
    args = argparser()
    paths = glob.glob(os.path.join(args.path,'sim_*'))
    model_types = ['dphpg','dphprg']
    models = []
    for path in paths:
        for model_type in model_types:
            mm = glob.glob(os.path.join(path, model_type, 'results*.db'))
            for m in mm:
                models.append((model_type, m))

    pplrs = []
    for model in models:
        print('Processing Model {}'.format(model[0]), end = ' ')
        try:
            pplrs.append(ppl_generation(model))
            print('Passed')
        except pd.io.sql.DatabaseError:
            print('Failed')
            pass

    df = pd.DataFrame(pplrs, columns = ('Type', 'Scenario', 'PPL_Linf','ES_Linf'))
    df.to_csv(os.path.join(os.path.split(args.path)[0], 'post_pred_loss_results.csv'), index = False)

# EOF
