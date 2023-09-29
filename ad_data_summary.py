from data import MixedData
from models import Results
from glob import glob
from samplers import bincount2D_vectorized
import pandas as pd
import numpy as np
import os

EPS = np.finfo(float).eps

# Rank
rank = {
    'cardio' : {
        'source'    : './ad/cardio/rank_data.csv',
        'outcome'   : './ad/cardio/rank_outcome.csv',
        'realtype'  : 'rank',
        'cats'      : '[19,20,21]',
        'model'     : 'pypprgln',
        'model_radius' : 'True',
        },
    'cover' : {
        'source'    : './ad/cover/rank_data.csv',
        'outcome'   : './ad/cover/rank_outcome.csv',
        'realtype'  : 'rank',
        'cats'      : '[9,10,11,12]',
        'model'     : 'pypprgln',
        'model_radius' : 'True',
        },
    'mammography' : {
        'source'    : './ad/mammography/rank_data.csv',
        'outcome'   : './ad/mammography/rank_outcome.csv',
        'realtype'  : 'rank',
        'cats'      : '[6,7,8]',
        'model'     : 'pypprgln',
        'model_radius' : 'True',
        },
    'pima' : {
        'source'    : './ad/pima/rank_data.csv',
        'outcome'   : './ad/pima/rank_outcome.csv',
        'realtype'  : 'rank',
        'cats'      : '[8,9,10,11,12]',
        'model'     : 'pypprgln',
        'model_radius' : 'True',
        },
    'annthyroid' : {
        'source'    : './ad/annthyroid/rank_data.csv',
        'outcome'   : './ad/annthyroid/rank_outcome.csv',
        'realtype'  : 'rank',
        'cats'      : '[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]',
        'model'     : 'pypprgln',
        'model_radius' : 'True',
        },
    'yeast' : {
        'source'    : './ad/yeast/rank_data.csv',
        'outcome'   : './ad/yeast/rank_outcome.csv',
        'realtype'  : 'rank',
        'cats'      : '[6,7]',
        'model'     : 'pypprgln',
        'model_radius' : 'True',
        },
    }
rank_datasets = ['annthyroid','cardio','cover','mammography','pima','yeast']
# Real
real = {
    'cardio' : {
        'source'    : './ad/cardio/real_data.csv',
        'outcome'   : './ad/cardio/real_outcome.csv',
        'quantile'  : '0.85',
        'cats'      : '[15,16,17,18,19,20,21,22,23,24]',
        'decluster' : 'False',
        'model'     : 'pypprgln',    
        },
    'cover' : {
        'source'    : './ad/cover/real_data.csv',
        'outcome'   : './ad/cover/real_outcome.csv',
        'quantile'  : '0.98',
        'cats'      : '[9,10,11,12]',
        'decluster' : 'False',
        'model'     : 'pypprgln',
        },
    'mammography' : {
        'source'    : './ad/mammography/real_data.csv',
        'outcome'   : './ad/mammography/real_outcome.csv',
        'quantile'  : '0.95',
        'cats'      : '[5,6,7,8,9]',
        'decluster' : 'False',
        'model'     : 'pypprgln',
        },
    'pima' : {
        'source'    : './ad/pima/real_data.csv',
        'outcome'   : './ad/pima/real_outcome.csv',
        'quantile'  : '0.90',
        'cats'      : '[7,8,9,10,11,12]',
        'decluster' : 'False',
        'model'     : 'pypprgln',
        },
    'annthyroid' : {
        'source'    : './ad/annthyroid/real_data.csv',
        'outcome'   : './ad/annthyroid/real_outcome.csv',
        'quantile'  : '0.85',
        'cats'      : '[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]',
        'decluster' : 'False',
        'model'     : 'pypprgln',
        },
    'yeast' : {
        'source'    : './ad/yeast/real_data.csv',
        'outcome'   : './ad/yeast/real_outcome.csv',
        'quantile'  : '0.90',
        'cats'      : '[4,5,8,9,10]',
        'decluster' : 'False',
        'model'     : 'pypprgln',
        },
    }
real_datasets = ['annthyroid','cardio','cover','mammography','pima','yeast']
# Cat
cat = {
    'cover' : {
        'source'    : './ad/cover/cat_data.csv',
        'outcome'   : './ad/cover/cat_outcome.csv',
        'model'     : 'pypprgln',
        'cats'      : '[0,1,2,3,4,5,6,7,8,9]',
        },
    'pima' : {
        'source'    : './ad/pima/cat_data.csv',
        'outcome'   : './ad/pima/cat_outcome.csv',
        'model'     : 'pypprgln',
        'cats'      : '[0,1,2,3,4,5,6,7]',
        },
    'yeast' : {
        'source'    : './ad/yeast/cat_data.csv',
        'outcome'   : './ad/yeast/cat_outcome.csv',
        'model'     : 'pypprgln',
        'cats'      : '[0,1,2,3,4,5,6,7]',
        },
    'solarflare' : {
        'source'    : './ad/solarflare/cat_data.csv',
        'outcome'   : './ad/solarflare/cat_outcome.csv',
        'model'     : 'pypprgln',
        'cats'      : '[0,1,2,3,4,5,6,7,8,9]',
        },
    }
cat_datasets = ['cover','pima','solarflare','yeast']

# out = {
#     'name'       : dataset['name'],
#     'quantile'   : dataset['quantile'],
#     'n_over_threshold' : dat.nDat,
#     'n_over_anom' : dat.Y.sum(),
#     'total_cols' : dat.nCol + dat.nCat,
#     'prevalence' : dat.Y.sum() / dat.nDat,
#     'raw_prevalence' : y.sum() / y.shape[0],
#     'N_raw_obsv' : y.shape[0],
#     'N_raw_anom' : y.sum(),
#     'P_anom_pres' : dat.Y.sum() / y.sum(),
#     'P_over_thre' : dat.nDat / y.shape[0],
#     }
# out['OR_keep_anom'] = (
#     out['P_anom_pres'] / (1 - out['P_anom_pres'] + EPS)
#     / (out['P_over_thre'] / (1 - out['P_over_thre'] + EPS))
#     )

summaries = []

for dataset in rank_datasets:
    raw = pd.read_csv(rank[dataset]['source']).values
    raw = raw[~np.isnan(raw).any(axis = 1)]
    out = pd.read_csv(rank[dataset]['outcome']).values.ravel()
    out = out[~np.isnan(out)]
    
    dat = MixedData(raw = raw, cat_vars = eval(rank[dataset]['cats']), realtype = 'rank')
    dat.fill_outcome(out)
    
    # verify
    assert (raw.shape[0] == out.shape[0])
    
    desc = {
        'name'     : dataset,
        'regime'   : 'rank',
        'path'     : rank[dataset]['source'],
        'raw_cols' : raw.shape[1],
        'mod_cols' : dat.nCol + dat.nCat + 1,
        'real_vars': dat.nCol,
        'cat_vars' : raw.shape[1] - dat.nCol,
        'cat_cols' : dat.nCat,
        'N_raw'    : out.shape[0],
        'N_over'   : dat.nDat,
        'A_raw'    : out.sum(),
        'A_over'   : dat.Y.sum(),
        }
    summaries.append(desc)

for dataset in real_datasets:
    raw = pd.read_csv(real[dataset]['source']).values
    raw = raw[~np.isnan(raw).any(axis = 1)]
    out = pd.read_csv(real[dataset]['outcome']).values.ravel()
    out = out[~np.isnan(out)]
    
    dat = MixedData(raw = raw, cat_vars = eval(real[dataset]['cats']), realtype = 'threshold')
    dat.fill_outcome(out)

    # verify
    assert (raw.shape[0] == out.shape[0])
    
    desc = {
        'name'     : dataset,
        'regime'   : 'threshold',
        'path'     : real[dataset]['source'],
        'quantile' : real[dataset]['quantile'],
        'raw_cols' : raw.shape[1],
        'mod_cols' : dat.nCol + dat.nCat,
        'real_vars': dat.nCol,
        'cat_vars' : raw.shape[1] - dat.nCol,
        'cat_cols' : dat.nCat,
        'N_raw'    : out.shape[0],
        'N_over'   : dat.nDat,
        'A_raw'    : out.sum(),
        'A_over'   : dat.Y.sum(),
        }
    summaries.append(desc)

for dataset in cat_datasets:
    raw = pd.read_csv(cat[dataset]['source']).values
    raw = raw[~np.isnan(raw).any(axis = 1)]
    out = pd.read_csv(cat[dataset]['outcome']).values.ravel()
    out = out[~np.isnan(out)]
    
    dat = MixedData(
        raw = raw, cat_vars = eval(cat[dataset]['cats']), realtype = 'rank',
        )
    dat.fill_outcome(out)

    # verify
    assert (raw.shape[0] == out.shape[0])
    
    desc = {
        'name'     : dataset,
        'regime'   : 'categorical',
        'path'     : cat[dataset]['source'],
        'raw_cols' : raw.shape[1],
        'mod_cols' : dat.nCol + dat.nCat,
        'real_vars': dat.nCol,
        'cat_vars' : raw.shape[1] - dat.nCol,
        'cat_cols' : dat.nCat,
        'N_raw'    : out.shape[0],
        'N_over'   : dat.nDat,
        'A_raw'    : out.sum(),
        'A_over'   : dat.Y.sum(),
        }
    summaries.append(desc)


summary_df = pd.DataFrame(summaries)
summary_df.to_csv('./ad/data_summary_updated.csv', index = False)

# basepath = './ad'
# folders = ['cardio','cover','mammography','annthyroid','yeast']
# result_paths = []
# for folder in folders:
#     for file in glob(os.path.join(basepath, folder, 'results_xv*')):
#         result_paths.append(file)

# result_summaries = []
# for path in result_paths:
#     result = Results['mpypprgln'](path)

#     avg_cluster_count =  (
#         bincount2D_vectorized(
#             result.samples.delta, 
#             result.max_clust_count + 1,
#             ) > 0
#         ).sum(axis = 1).mean()
#     concentration = result.GEMPrior.concentration
#     discount = result.GEMPrior.discount

#     out = {
#         'path' : path,
#         'avg_cluster_count' : avg_cluster_count,
#         'discount' : discount,
#         'concentration' : concentration,
#         'n_train' : result.nDat,
#         }
#     result_summaries.append(out)

# res_df = pd.DataFrame(result_summaries)
# res_df.to_csv('./ad/result_summary.csv', index = False)
    









# EOF
