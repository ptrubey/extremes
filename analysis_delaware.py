from __future__ import division
import sys, os, glob, re
import numpy as np, pandas as pd, pickle as pkl
from io import BytesIO
from geopy.distance import geodesic

from energy import limit_cpu
from data import Data_From_Raw

import model_spypprg as mc
import varbayes as vb
import model_spypprgr as mcr
import posterior as post

dataset = 'del'

data_in_base   = './datasets/slosh/slosh_{}_data.csv.gz'
alpha_out_base = './datasets/slosh/{}/{}_alphas.csv.gz'
clust_out_base = './datasets/slosh/{}/{}_clusters.csv.gz'
delta_out_base = './datasets/slosh/{}/{}_delta.csv.gz'
graph_out_base = './datasets/slosh/{}/{}_graph.csv.gz'
slosh_out_base = './datasets/slosh/{}/{}_locinfo.csv.gz'
theta_out_base = './datasets/slosh/{}/{}_theta.csv.gz'

args = {
    'dataset'   : 'del',
    'quantile'  : 0.90, 
    'conc'      : 0.1,
    'disc'      : 0.1,
    }

def run_slosh_vb(
        dataset   : str,
        quantile  : float, 
        conc      : float, 
        disc      : float,
        ):
    data_in_path = data_in_base.format(dataset)
    alpha_out_path = alpha_out_base.format(dataset, 'vb')
    delta_out_path = delta_out_base.format(dataset, 'vb')
    clust_out_path = clust_out_base.format(dataset, 'vb')
    graph_out_path = graph_out_base.format(dataset, 'vb')
    slosh_out_path = slosh_out_base.format(dataset, 'vb')

    slosh = pd.read_csv(data_in_path, compression = 'gzip')
    slosh_ids = slosh.T[:8].T
    slosh_obs = slosh.T[8:].values.astype(np.float64)
    
    data = Data_From_Raw(slosh_obs, decluster = False, quantile = quantile)
    model = vb.VarPYPG(data, eta = conc, discount = disc)
    model.fit_advi()
    
    deltas = model.generate_conditional_posterior_deltas()
    alphas = model.generate_conditional_posterior_alphas()

    smat   = post.similarity_matrix(deltas)
    graph  = pd.DataFrame(post.minimum_spanning_trees(deltas)).rename(
        columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'},
        )
    pd.DataFrame(
        deltas.T, 
        columns = ['X{:03}'.format(i) for i in range(data.nDat)],
        ).to_csv(delta_out_path, index = False, compression = 'gzip')
    graph.to_csv(graph_out_path, index = False, compression = 'gzip')
    alphaM = np.empty((alphas.shape[0], alphas.shape[1], alphas.shape[2] + 1))
    alphaM[:,:,1:] = alphas
    alphaM[:,:,0]  = np.arange(alphas.shape[0]).reshape(alphas.shape[0], 1)
    pd.DataFrame(
        alphaM.reshape(-1,alphaM.shape[2]), 
        columns = ['Obs'] + ['A{:03}'.format(i) for i in range(alphas.shape[2])]
        ).to_csv(alpha_out_path, index = False, compression = 'gzip')
    pd.DataFrame(
        smat,
        columns = ['S{:03}'.format(i) for i in range(data.nDat)]
        ).to_csv(clust_out_path, index = False, compression = 'gzip')
    
    params = pd.read_csv('./datasets/slosh/slosh_params.csv').loc[data.I]
    params.to_csv(slosh_out_path, index = False, compression = 'gzip')
    return

def run_slosh_mc(
        dataset   : str,
        quantile  : float, 
        conc      : float, 
        disc      : float,
        ):
    data_in_path = data_in_base.format(dataset)
    alpha_out_path = alpha_out_base.format(dataset, 'mc')
    delta_out_path = delta_out_base.format(dataset, 'mc')
    clust_out_path = clust_out_base.format(dataset, 'mc')
    graph_out_path = graph_out_base.format(dataset, 'mc')
    slosh_out_path = slosh_out_base.format(dataset, 'mc')

    slosh = pd.read_csv(data_in_path, compression = 'gzip')
    slosh_ids = slosh.T[:8].T
    slosh_obs = slosh.T[8:].values.astype(np.float64)
    
    data = Data_From_Raw(slosh_obs, decluster = False, quantile = quantile)
    model = mc.Chain(data, concentration = conc, discout = disc)
    model.sample(50000, verbose = True)
    out = BytesIO()
    model.write_to_disk(out, 40001, 10)
    res = mc.Result(out)

    deltas = res.samples.delta
    alphas = res.generate_conditional_posterior_predictive_zetas()
    
    smat   = post.similarity_matrix(deltas.T)
    graph  = pd.DataFrame(post.minimum_spanning_trees(deltas)).rename(
        columns = {0 : 'node1', 1 : 'node2', 2 : 'node3'},
        )
    pd.DataFrame(
        deltas, 
        columns = ['X{:03}'.format(i) for i in range(data.nDat)],
        ).to_csv(delta_out_path, index = False, compression = 'gzip')
    graph.to_csv(graph_out_path, index = False, compression = 'gzip')
    alphaM = np.empty((alphas.shape[0], alphas.shape[1], alphas.shape[2] + 1))
    alphaM[:,:,1:] = alphas
    alphaM[:,:,0]  = np.arange(alphas.shape[0]).reshape(alphas.shape[0], 1)
    pd.DataFrame(
        alphaM.reshape(-1,alphaM.shape[2]), 
        columns = ['A{:03}'.format(i) for i in range(alphas.shape[2])]
        ).to_csv(alpha_out_path, index = False)
    pd.DataFrame(
        smat,
        columns = ['S{:03}'.format(i) for i in range(data.nDat)]
        ).to_csv(clust_out_path, index = False, compression = 'gzip')
    params = pd.read_csv('./datasets/slosh/slosh_params.csv').loc[data.I]
    params.to_csv(slosh_out_path, index = False, compression = 'gzip')
    return

def run_slosh_reg(
        dataset   : str,
        quantile  : float, 
        conc      : float, 
        disc      : float,
        ):
    data_in_path = data_in_base.format(dataset)
    alpha_out_path = alpha_out_base.format(dataset, 'reg')
    delta_out_path = delta_out_base.format(dataset, 'reg')
    clust_out_path = clust_out_base.format(dataset, 'reg')
    graph_out_path = graph_out_base.format(dataset, 'reg')
    theta_out_path = theta_out_base.format(dataset, 'reg')
    slosh_out_path = slosh_out_base.format(dataset, 'reg')
    
    params = pd.read_csv('./datasets/slosh/slosh_params.csv')
    locats = pd.read_csv('./datasets/slosh/slosh_locs.csv')[['IDX','NearOcean','elevation']]
    locats.IDX = locats.IDX.astype(np.int64)
    
    slosh  = pd.read_csv(data_in_path, compression = 'gzip')
    sloshltd_ids = slosh.iloc[:,:8]
    sloshltd_idi = sloshltd_ids.set_index('IDX').join(locats.set_index('IDX')).reset_index()
    sloshltd_idi.elevation = sloshltd_idi.elevation / 10 # decafeet
    sloshltd_obs = slosh.iloc[:,8:].T.values.astype(np.float64)

    params.loc[params.theta < 100, 'theta'] += 360
    params_par = mcr.summarize(params.values)
    params_std = mcr.scale(params.values, params_par)
    
    locatx_par = mcr.summarize(sloshltd_ids[['long','lat']].values)
    locatx_std = mcr.scale(sloshltd_ids[['long','lat']].values, locatx_par)
    
    x_observation = params_std
    x_location    = np.hstack((
        locatx_std, 
        sloshltd_idi[['NearOcean','elevation']].values.astype(np.float64),
        ))

    storm_locs = list(map(tuple, params[['lat','long']].values))
    slosh_locs = list(map(tuple, sloshltd_ids[['lat','long']].values))
    distances  = np.array([
        [geodesic(slosh_loc, storm_loc).miles for slosh_loc in slosh_locs] 
        for storm_loc in storm_locs
        ])
    # storms on the horizontal, locations on the vertical.
    x_interaction = distances[:,:,None] / 100 # in hectomiles.
    data = mcr.RegressionData(
            raw_real    = sloshltd_obs, 
            real_type   = 'threshold',
            decluster   = False, 
            quantile    = quantile,
            observation = x_observation,
            location    = x_location,
            interaction = x_interaction,
            )
    model = mcr.Chain(
        data, 
        p = 10, 
        concentration = conc, 
        discount = disc,
        )
    model.sample(50000, verbose = True)
    out = BytesIO()
    model.write_to_disk(out, 40000, 10)
    res = mcr.Result(out)

    deltas = res.samples.delta
    thetas = res.samples.theta
    alphas = res.generate_conditional_posterior_predictive_alphas()

    raise

    alphaM = np.empty((alphas.shape[0], alphas.shape[1], alphas.shape[2] + 1))
    alphaM[:,:,1:] = alphas
    alphaM[:,:,0]  = np.arange(alphas.shape[0]).reshape(alphas.shape[0], 1)
    
    smat   = post.similarity_matrix(deltas)
    graph  = pd.DataFrame(post.minimum_spanning_trees(deltas)).rename(
        columns = {0 : 'node1', 1 : 'node2', 2 : 'node3'},
        )
    
    pd.DataFrame(
        alphaM.reshape(-1,alphas.shape[2]), 
        columns = ['A{:03}'.format(i) for i in range(alphas.shape[2])]
        ).to_csv(alpha_out_path, index = False)
    pd.DataFrame(deltas).to_csv(
        delta_out_path, index = False, compression = 'gzip',
        )
    pd.DataFrame(alphas).to_csv(
        alpha_out_path, index = False, compression = 'gzip',
        )
    pd.DataFrame(thetas).to_csv(
        theta_out_path, index = False, compression = 'gzip',
        )
    graph.to_csv(graph_out_path, index = False, compression = 'gzip')
    pd.DataFrame(
        smat,
        columns = ['S{:03}'.format(i) for i in range(data.nDat)]
        ).to_csv(clust_out_path, index = False, compression = 'gzip')
    slosh_ids.loc[data.I].to_csv(
        slosh_out_path, index = False, compression = 'gzip',
        )
    
if __name__ == '__main__':
    # run_slosh_vb(**args)
    # run_slosh_mc(**args)
    run_slosh_reg(**args)

# EOF