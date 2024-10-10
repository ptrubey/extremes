import argparse, os, pickle
import pandas as pd, numpy as np
from geopy.distance import geodesic

from model_spypprgr import RegressionData, Chain, Result, summarize, scale
from energy import limit_cpu

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('conc')
    p.add_argument('disc')
    # p.add_argument('in_data_path')
    # p.add_argument('in_regressors_path')
    # p.add_argument('in_location_path')
    # p.add_argument('out_path')
    p.add_argument('--fixed_effects', default = 'True')
    p.add_argument('--nSamp', default = '30000')
    p.add_argument('--nKeep', default = '15000')
    p.add_argument('--nThin', default = '15')
    p.add_argument('--verbose', default = False)
    
    # p.add_argument('--realtype', default = 'threshold')
    # p.add_argument('--quantile', default = 0.95)
    # p.add_argument('--decluster', default = 'True')
    # p.add_argument('--p', default = '10')
    # p.add_argument('--maxclust', default = '200')
    # p.add_argument('conc', default = '1e-1')
    # p.add_argument('disc', default = '1e-1')

    return p.parse_args()

class Heap(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        return

def fake_argparser():
    p = Heap(
        src = 'apt',
        conc = 0.01,
        disc = 0.01,
        nSamp = 30000,
        nKeep = 15000,
        nThin = 15,
        verbose = True,
        )
    return p

basepath = './datasets/slosh/slosh_{}_data.csv.gz'

if __name__ == '__main__':
    args = argparser()
    # args = fake_argparser()
    limit_cpu()

    params = pd.read_csv('./datasets/slosh/slosh_params.csv')
    locats = pd.read_csv('./datasets/slosh/slosh_locs.csv')[['IDX','NearOcean','elevation']]
    locats.IDX = locats.IDX.astype(np.int64)
    slosh  = pd.read_csv(basepath.format(args.src), compression = 'gzip')

    sloshltd_ids = slosh.iloc[:,:8]
    sloshltd_idi = sloshltd_ids.set_index('IDX').join(locats.set_index('IDX')).reset_index()
    sloshltd_idi.elevation = sloshltd_idi.elevation / 10 # decafeet
    sloshltd_obs = slosh.iloc[:,8:].T.values.astype(np.float64)

    params.loc[params.theta < 100, 'theta'] += 360
    params_par = summarize(params.values)
    params_std = scale(params.values, params_par)
    
    try:
        locatx_par = summarize(sloshltd_ids[['x','y']].values)
        locatx_std = scale(sloshltd_ids[['long','lat']].values, locatx_par)
    except KeyError:
        locatx_par = summarize(sloshltd_ids[['long','lat']].values)
        locatx_std = scale(sloshltd_ids[['long','lat']].values, locatx_par)
    
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
    data = RegressionData(
            raw_real    = sloshltd_obs, 
            real_type   = 'threshold',
            decluster   = False, 
            quantile    = 0.90,
            observation = x_observation,
            location    = x_location,
            interaction = x_interaction,
            )
    model = Chain(
        data, 
        p = 10, 
        concentration = float(args.conc), 
        discount = float(args.disc),
        fixed_effects = eval(args.fixed_effects)
        )
    model.sample(int(args.nSamp), verbose = args.verbose)
    fname = 'slosh_{}_{}_{}_{}'.format(
        args.src, args.conc, args.disc, int(eval(args.fixed_effects))
        )
    fname_pkl = fname + '.pkl'
    fpath_pkl = os.path.join('./test/test', fname_pkl)
    model.write_to_disk(fpath_pkl, int(args.nKeep), int(args.nThin))
    fpath_del = os.path.join('./test/test', fname + '_delta.csv')
    fpath_gam = os.path.join('./test/test', fname + '_gamma.pkl')
      
    res = Result(fpath_pkl)
    postdeltas = res.samples.delta
    postalphas = res.generate_conditional_posterior_predictive_gammas()

    pd.DataFrame(postdeltas).to_csv(fpath_del, index = False)
    with open(fpath_gam, 'wb') as file:
        pickle.dump(postalphas, file)

# EOF