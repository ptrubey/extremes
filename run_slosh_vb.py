from __future__ import division
import sys, os, glob, re
import numpy as np
import pandas as pd
import multiprocessing as mp
import sqlite3 as sql
from io import BytesIO
from time import sleep
from collections import namedtuple
from numpy.random import uniform
import pickle as pkl

from energy import limit_cpu, postpred_loss_full, energy_score_full_sc
from data import Data_From_Sphere, Data_From_Raw
import varbayes as vb
import posterior as post

raw_path  = './datasets/slosh/filtered_data.csv.gz'
out_sql   = './datasets/slosh/results.sql'
out_table = 'energy'

def run_slosh(
        path_in,   path_out, 
        delta_out, cluster_out, 
        quantile, 
        eta,       discount,
        ):
    slosh = pd.read_csv(path_in, compression = 'gzip')
    slosh_ids = slosh.T[:8].T
    slosh_obs = slosh.T[8:].values.astype(np.float64)
    data = Data_From_Raw(slosh_obs, decluster = False, quantile = quantile)
    model = vb.VarPYPG(data, eta = eta, discount = discount)
    model.fit_advi()
    deltas = model.generate_conditional_posterior_deltas()
    smat = post.similarity_matrix(deltas)
    graph = post.minimum_spanning_trees(smat)
    g     = pd.DataFrame(graph)
    g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
    d = {
        'ids'    : slosh_ids,
        'obs'    : slosh_obs,
        'deltas' : deltas,
        'smat'   : smat,
        'graph'  : g,
        }
    with open(path_out, 'wb') as file:
        pkl.dump(d, file)
    pd.DataFrame({'delta' : deltas}).to_csv(delta_out, index = False)
    pd.DataFrame({
        'obs' : np.arange(model.N),
        'pre' : post.emergent_clusters_pre(model),
        'pos' : post.emergent_clusters_post(model),
        }).to_csv(cluster_out, index = False)
    return

def run_slosh_for_nclusters(data, dataname, concentrations, discounts):
    ClusterCount = namedtuple('ClusterCount', 'eta discount ncluster')
    counts = []
    for eta in concentrations:
        for discount in discounts:
            model = vb.VarPYPG(data, eta = eta, discount = discount)
            model.fit_advi()
            nclusters = np.unique(post.emergent_clusters_post(model)).shape[0]
            counts.append(ClusterCount(eta, discount, nclusters))
    pd.DataFrame(counts).to_csv('./test/ncluster_{}.csv'.format(dataname))
    return

def instantiate_data(path, quantile):
    slosh = pd.read_csv(path, compression = 'gzip')
    slosh_obs = slosh.T[8:].values.astype(np.float64)
    return Data_From_Raw(slosh_obs, decluster = False, quantile = quantile)

run = {
    't90' : False, # True,
    'ltd' : False, # True,
    'xpt' : False, # True,
    'apt' : False, # True,
    'emg' : False, # True,
    'loc' : True,
    'del' : True,
    'nyc' : True,
    }
path_in_base  = './datasets/slosh/slosh_{}_data.csv.gz'
path_out_base = './datasets/slosh/slosh_{}.pkl'
clus_out_base = './datasets/slosh/slosh_{}_clusters.csv'
delt_out_base = './datasets/slosh/slosh_{}_delta.csv'
args = {
    't90' : {
        'path_in'     : path_in_base.format('t90'),
        'path_out'    : path_out_base.format('t90'),
        'cluster_out' : clus_out_base.format('t90'),
        'delta_out'   : delt_out_base.format('t90'),
        'quantile'    : 0.90,
        },
    'ltd' : {
        'path_in'     : path_in_base.format('ltd'),
        'path_out'    : path_out_base.format('ltd'),
        'cluster_out' : clus_out_base.format('ltd'),
        'delta_out'   : delt_out_base.format('ltd'),
        'quantile'     : 0.95, 
        },
    'xpt' : {
        'path_in'     : path_in_base.format('xpt'),
        'path_out'    : path_out_base.format('xpt'),
        'cluster_out' : clus_out_base.format('xpt'),
        'delta_out'   : delt_out_base.format('xpt'),
        'quantile'     : 0.95, 
        },
    'apt' : {
        'path_in'     : path_in_base.format('apt'),
        'path_out'    : path_out_base.format('apt'),
        'cluster_out' : clus_out_base.format('apt'),
        'delta_out'   : delt_out_base.format('apt'),
        'quantile'     : 0.95, 
        },
    'emg' : {
        'path_in'     : path_in_base.format('emg'),
        'path_out'    : path_out_base.format('emg'),
        'cluster_out' : clus_out_base.format('emg'),
        'delta_out'   : delt_out_base.format('emg'),
        'quantile'     : 0.95, 
        },
    'loc' : {
        'path_in'     : path_in_base.format('loc'),
        'path_out'    : path_out_base.format('loc'),
        'cluster_out' : clus_out_base.format('loc'),
        'delta_out'   : delt_out_base.format('loc'),
        'quantile'     : 0.95, 
        },
    'del' : {
        'path_in'     : path_in_base.format('del'),
        'path_out'    : path_out_base.format('del'),
        'cluster_out' : clus_out_base.format('del'),
        'delta_out'   : delt_out_base.format('del'),
        'quantile'     : 0.95, 
        },
    'nyc' : {
        'path_in'     : path_in_base.format('nyc'),
        'path_out'    : path_out_base.format('nyc'),
        'cluster_out' : clus_out_base.format('nyc'),
        'delta_out'   : delt_out_base.format('nyc'),
        'quantile'     : 0.95, 
        },
    }

if __name__ == '__main__':
    limit_cpu()
    # for dataset in run:
    #     if run[dataset]:
    #         run_slosh(**args[dataset])

    concs = [0.001, 0.01, 0.1, 0.2]
    discs = [0.001, 0.01, 0.1, 0.2]

    for dataset in run.keys(): # ['ltd','xpt','apt','emg']:
        if run[dataset]:
            data = instantiate_data(
                args[dataset]['path_in'], args[dataset]['quantile'],
                )
            run_slosh_for_nclusters(data, dataset, concs, discs)

    if run['t90']:
        data = instantiate_data(
            args['t90']['path_in'], args['t90']['quantile'],
            )
        run_slosh_for_nclusters(data, 't90', concs[2:], discs[2:])
    
    if False: # sloshapt
        sloshapt = pd.read_csv(
            './datasets/slosh/slosh_apt_data.csv.gz', 
            compression = 'gzip',
            )
        sloshapt_ids = sloshapt.T[:8].T
        sloshapt_obs = sloshapt.T[8:].values.astype(np.float64)
        
        data = Data_From_Raw(sloshapt_obs, decluster = False, quantile = 0.95)
        model = vb.VarPYPG(data)
        model.fit_advi()
        postalphas = model.generate_conditional_posterior_alphas()

        inputs = pd.read_csv('~/git/surge/data/inputs.csv')
        finputs = inputs.iloc[model.data.I]

        deltas = model.generate_conditional_posterior_deltas()
        import posterior as post
        smat   = post.similarity_matrix(deltas)
        graph  = post.minimum_spanning_trees(smat)
        g      = pd.DataFrame(graph)
        g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
        # write to disk

        d = {
            'ids'    : sloshapt_ids,
            'obs'    : sloshapt_obs,
            'inputs' : finputs,
            'alphas' : postalphas,
            'deltas' : deltas,
            'smat'   : smat,
            'graph'  : g,
            }
        with open('./datasets/slosh/sloshapt.pkl', 'wb') as file:
            pkl.dump(d, file)
        g.to_csv('./datasets/slosh/sloshapt_mst.csv', index = False)
        pd.DataFrame(deltas).to_csv('./datasets/slosh/sloshapt_delta.csv', index = False)
        
        pd.DataFrame({
            'obs' : np.arange(model.N),
            'pre' : post.emergent_clusters_pre(model),
            'pos' : post.emergent_clusters_post(model),
            }).to_csv('./datasets/slosh/sloshapt_clusters.csv', index = False)

    if False: # sloshltd
        sloshltd = pd.read_csv(
            './datasets/slosh/slosh_ltd_data.csv.gz', 
            compression = 'gzip',
            )
        sloshltd_ids = sloshltd.T[:8].T
        sloshltd_obs = sloshltd.T[8:].values.astype(np.float64)

        data = Data_From_Raw(sloshltd_obs, decluster = False, quantile = 0.95)
        model = vb.VarPYPG(data)
        model.fit_advi()
        postalphas = model.generate_conditional_posterior_alphas()

        inputs = pd.read_csv('~/git/surge/data/inputs.csv')
        finputs = inputs.iloc[model.data.I]

        deltas = model.generate_conditional_posterior_deltas()
        import posterior as post
        smat   = post.similarity_matrix(deltas)
        graph  = post.minimum_spanning_trees(smat)
        g      = pd.DataFrame(graph)
        g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
        # write to disk

        d = {
            'ids'    : sloshltd_ids,
            'obs'    : sloshltd_obs,
            'alphas' : postalphas,
            'inputs' : finputs,
            'deltas' : deltas,
            'smat'   : smat,
            'graph'  : g,
            }
        with open('./datasets/slosh/sloshltd.pkl', 'wb') as file:
            pkl.dump(d, file)
        g.to_csv('./datasets/slosh/sloshltd_mst.csv', index = False)
        pd.DataFrame(deltas).to_csv('./datasets/slosh/sloshltd_delta.csv', index = False)
        
        pd.DataFrame({
            'obs' : np.arange(model.N),
            'pre' : post.emergent_clusters_pre(model),
            'pos' : post.emergent_clusters_post(model),
            }).to_csv('./datasets/slosh/sloshltd_clusters.csv', index = False)

    if False: # slosht90
        slosht90 = pd.read_csv(
            './datasets/slosh/slosh_t90_data.csv.gz', 
            compression = 'gzip',
            )
        slosht90_ids = slosht90.T[:8].T
        slosht90_obs = slosht90.T[8:].values.astype(np.float64)

        data = Data_From_Raw(slosht90_obs, decluster = False, quantile = 0.90)
        model = vb.VarPYPG(data)
        model.fit_advi()
        postalphas = model.generate_conditional_posterior_alphas()

        inputs = pd.read_csv('~/git/surge/data/inputs.csv')
        finputs = inputs.iloc[model.data.I]

        deltas = model.generate_conditional_posterior_deltas()
        import posterior as post
        smat   = post.similarity_matrix(deltas)
        graph  = post.minimum_spanning_trees(smat)
        g      = pd.DataFrame(graph)
        g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
        # write to disk

        d = {
            'ids'    : slosht90_ids,
            'obs'    : slosht90_obs,
            'inputs' : finputs,
            'alphas' : postalphas,
            'deltas' : deltas,
            'smat'   : smat,
            'graph'  : g,
            }
        with open('./datasets/slosh/slosht90.pkl', 'wb') as file:
            pkl.dump(d, file)
        g.to_csv('./datasets/slosh/slosht90_mst.csv', index = False)
        pd.DataFrame(deltas).to_csv('./datasets/slosh/slosht90_delta.csv', index = False)
        
        pd.DataFrame({
            'obs' : np.arange(model.N),
            'pre' : post.emergent_clusters_pre(model),
            'pos' : post.emergent_clusters_post(model),
            }).to_csv('./datasets/slosh/slosht90_clusters.csv', index = False)

    if False: # sloshemg
        sloshemg = pd.read_csv(
            './datasets/slosh/slosh_emg_data.csv.gz', 
            compression = 'gzip',
            )
        sloshemg_ids = sloshemg.T[:8].T
        sloshemg_obs = sloshemg.T[8:].values.astype(np.float64)
        
        data = Data_From_Raw(sloshemg_obs, decluster = False, quantile = 0.95)
        model = vb.VarPYPG(data)
        model.fit_advi()
        postalphas = model.generate_conditional_posterior_alphas()

        inputs = pd.read_csv('~/git/surge/data/inputs.csv')
        finputs = inputs.iloc[model.data.I]

        deltas = model.generate_conditional_posterior_deltas()
        import posterior as post
        smat   = post.similarity_matrix(deltas)
        graph  = post.minimum_spanning_trees(smat)
        g      = pd.DataFrame(graph)
        g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
        # write to disk

        d = {
            'ids'    : sloshemg_ids,
            'obs'    : sloshemg_obs,
            'inputs' : finputs,
            'alphas' : postalphas,
            'deltas' : deltas,
            'smat'   : smat,
            'graph'  : g,
            }
        with open('./datasets/slosh/sloshemg.pkl', 'wb') as file:
            pkl.dump(d, file)
        g.to_csv('./datasets/slosh/sloshemg_mst.csv', index = False)
        pd.DataFrame(deltas).to_csv('./datasets/slosh/sloshemg_delta.csv', index = False)
        
        pd.DataFrame({
            'obs' : np.arange(model.N),
            'pre' : post.emergent_clusters_pre(model),
            'pos' : post.emergent_clusters_post(model),
            }).to_csv('./datasets/slosh/sloshemg_clusters.csv', index = False)

    if False: # sloshxpt
        sloshxpt = pd.read_csv(
            './datasets/slosh/slosh_xpt_data.csv.gz', 
            compression = 'gzip',
            )
        sloshxpt_ids = sloshxpt.T[:8].T
        sloshxpt_obs = sloshxpt.T[8:].values.astype(np.float64)
        
        data = Data_From_Raw(sloshxpt_obs, decluster = False, quantile = 0.95)
        model = vb.VarPYPG(data)
        model.fit_advi()
        postalphas = model.generate_conditional_posterior_alphas()

        inputs = pd.read_csv('~/git/surge/data/inputs.csv')
        finputs = inputs.iloc[model.data.I]

        deltas = model.generate_conditional_posterior_deltas()
        import posterior as post
        smat   = post.similarity_matrix(deltas)
        graph  = post.minimum_spanning_trees(smat)
        g      = pd.DataFrame(graph)
        g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
        # write to disk

        d = {
            'ids'    : sloshxpt_ids,
            'obs'    : sloshxpt_obs,
            'inputs' : finputs,
            'alphas' : postalphas,
            'deltas' : deltas,
            'smat'   : smat,
            'graph'  : g,
            }
        with open('./datasets/slosh/sloshxpt.pkl', 'wb') as file:
            pkl.dump(d, file)
        g.to_csv('./datasets/slosh/sloshxpt_mst.csv', index = False)
        pd.DataFrame(deltas).to_csv('./datasets/slosh/sloshxpt_delta.csv', index = False)
        
        pd.DataFrame({
            'obs' : np.arange(model.N),
            'pre' : post.emergent_clusters_pre(model),
            'pos' : post.emergent_clusters_post(model),
            }).to_csv('./datasets/slosh/sloshxpt_clusters.csv', index = False)

# EOF
