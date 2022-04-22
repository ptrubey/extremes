from argparser import argparser_ppl as argparser
import numpy as np
import pandas as pd
import sqlite3 as sql
import os
import glob
from numpy.random import gamma, choice
from collections import namedtuple
from postpred_loss import ResultFactory
from energy import limit_cpu

PPLResult = namedtuple('PPLResult', 'Type Scenario PPL_Linf ES_Linf PPL_Linf_F ES_Linf_F')

def ppl_generation(args):
    model, path = args
    result = ResultFactory(model, path)
    scenario = os.path.splitext(os.path.split(path)[1])[0]
    pplr = PPLResult(
            model[0],
            scenario,
            result.posterior_predictive_loss_Linf(),
            result.energy_score_Linf(),
            result.postpred_loss_Linf_full(),
            result.energy_score_Linf_full(),
            )
    return pplr

base_path = './simulated/sphere'

if __name__ == '__main__':
    limit_cpu()
    model_types = ['sdpppg', 'sdppprg', 'sdpppgln', 'sdppprgln']
    models = []

    for model_type in model_types:
        paths = glob.glob(os.path.join(base_path, 'results_{}_*.pkl'.format(model_type)))
        for path in paths:
            models.append((model_type, path))
    
    pplrs = []
    for model in models:
        print(('Processing Model {}'.format(model[0])).ljust(50), end = ' ')
        pplrs.append(ppl_generation(model))
        print('Passed!')
        # try:
        #     pplrs.append(ppl_generation(model))
        #     print('Passed')
        # except:
        #     print('Failed')
        #     pass

    df = pd.DataFrame(pplrs, columns = ('Type', 'Scenario', 'PPL_Linf','ES_Linf','PPL_Linf_F','ES_Linf_F'))
    df.to_csv(os.path.join(base_path, 'post_pred_loss_results.csv'), index = False)

# EOF
