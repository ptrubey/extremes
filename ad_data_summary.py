from data import MixedData
from models import Results
from glob import glob
from samplers import bincount2D_vectorized
import pandas as pd
import numpy as np
import os

EPS = np.finfo(float).eps

cardio = {
    'source'    : './ad/cardio/data.csv',
    'quantile'  : '0.85',
    'cats'      : '[15,16,17,18,19,20,21,22,23,24]',
    'name'      : 'cardio',
    }
cover = {
    'source'    : './ad/cover/data.csv',
    'quantile'  : '0.98',
    'cats'      : '[9,10,11,12]',
    'name'      : 'cover',
    }
mammography = {
    'source'    : './ad/mammography/data.csv',
    'quantile'  : '0.95',
    'cats'      : '[5,6,7,8,9]',
    'name'      : 'mammography',
    }
pima = {
    'source'    : './ad/pima/data.csv',
    'quantile'  : '0.90',
    'cats'      : '[7,8,9,10,11,12]',
    'name'      : 'pima',
    }
satellite = {
    'source'    : './ad/satellite/data.csv',
    'quantile'  : '0.95',
    'cats'      : '[36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]',
    'name'      : 'satellite',
    }
annthyroid = {
    'source'    : './ad/annthyroid/data.csv',
    'quantile'  : '0.85',
    'cats'      : '[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]',
    'name'      : 'annthyroid',
    }
yeast = {
    'source'    : './ad/yeast/data.csv',
    'quantile'  : '0.90',
    'cats'      : '[4,5,8,9,10]',
    'name'      : 'yeast',
    }

datasets = [cardio, cover, mammography, annthyroid, yeast]
summaries = []

for dataset in datasets:
    raw = pd.read_csv(dataset['source']).values
    raw = raw[~np.isnan(raw).any(axis = 1)]
    
    y = pd.read_csv(os.path.join(os.path.split(dataset['source'])[0], 'outcome.csv')).values.ravel()
    y = y[~np.isnan(y)]
    
    dat = MixedData(raw, eval(dataset['cats']), quantile = float(dataset['quantile']))
    dat.fill_outcome(y)

    out = {
        'name'       : dataset['name'],
        'quantile'   : dataset['quantile'],
        'n_over_threshold' : dat.nDat,
        'n_over_anom' : dat.Y.sum(),
        'total_cols' : dat.nCol + dat.nCat,
        'prevalence' : dat.Y.sum() / dat.nDat,
        'raw_prevalence' : y.sum() / y.shape[0],
        'N_raw_obsv' : y.shape[0],
        'N_raw_anom' : y.sum(),
        'P_anom_pres' : dat.Y.sum() / y.sum(),
        'P_over_thre' : dat.nDat / y.shape[0],
        }
    out['OR_keep_anom'] = (
        out['P_anom_pres'] / (1 - out['P_anom_pres'] + EPS)
        / (out['P_over_thre'] / (1 - out['P_over_thre'] + EPS))
        )
    summaries.append(out)

summary_df = pd.DataFrame(summaries)
summary_df.to_csv('./ad/data_summary.csv', index = False)

basepath = './ad'
folders = ['cardio','cover','mammography','annthyroid','yeast']
result_paths = []
for folder in folders:
    for file in glob(os.path.join(basepath, folder, 'results_xv*')):
        result_paths.append(file)

result_summaries = []
for path in result_paths:
    result = Results['mpypprgln'](path)

    avg_cluster_count =  (
        bincount2D_vectorized(
            result.samples.delta, 
            result.max_clust_count + 1,
            ) > 0
        ).sum(axis = 1).mean()
    concentration = result.GEMPrior.concentration
    discount = result.GEMPrior.discount

    out = {
        'path' : path,
        'avg_cluster_count' : avg_cluster_count,
        'discount' : discount,
        'concentration' : concentration,
        'n_train' : result.nDat,
        }
    result_summaries.append(out)

res_df = pd.DataFrame(result_summaries)
res_df.to_csv('./ad/result_summary.csv', index = False)
    









# EOF
