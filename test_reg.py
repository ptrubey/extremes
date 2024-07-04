import argparse, os, pickle
import pandas as pd, numpy as np

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
    p.add_argument('--nSamp', default = '30000')
    p.add_argument('--nKeep', default = '15000')
    p.add_argument('--nThin', default = '15')
    # p.add_argument('--realtype', default = 'threshold')
    # p.add_argument('--quantile', default = 0.95)
    # p.add_argument('--decluster', default = 'True')
    # p.add_argument('--p', default = '10')
    # p.add_argument('--maxclust', default = '200')
    # p.add_argument('conc', default = '1e-1')
    # p.add_argument('disc', default = '1e-1')

    return p.parse_args()

if __name__ == '__main__':
    args = argparser()
    limit_cpu()

    slosh  = pd.read_csv(
        './datasets/slosh/filtered_data.csv.gz', 
        compression = 'gzip',
        )
    sloshx = pd.read_csv('./datasets/slosh/slosh_params.csv')

    if args.src == 'ltd':
        sloshltd = ~slosh.MTFCC.isin(['C3061','C3081'])
    elif args.src == 'apt':
        sloshltd = slosh.MTFCC.isin(['K2451'])
    else:
        raise
    sloshltd_ids = slosh[sloshltd].iloc[:,:8]                             # location parms
    sloshltd_obs = slosh[sloshltd].iloc[:,8:].values.astype(np.float64).T # storm runs

    sloshx.loc[sloshx.theta < 100, 'theta'] += 360
    sloshx_par    = summarize(sloshx.values)
    locatx_par    = summarize(sloshltd_ids[['x','y']].values)
    sloshx_par.mean[-1] = locatx_par.mean[-1] # latitude values will be 
    sloshx_par.sd[-1]   = locatx_par.sd[-1]   # on same scale for both datasets
    sloshx_std    = scale(sloshx.values, sloshx_par)
    locatx_std    = scale(sloshltd_ids[['x','y']].values, locatx_par)
        
    x_observation = sloshx_std
    x_location    = locatx_std
    x_interaction = (sloshx_std[:,-1][None] * locatx_std[:,-1][:,None])[:,:,None]

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
        )
    model.sample(int(args.nSamp))
    fname = 'sloshltd_{}_{}_{}'.format(args.src, args.conc, args.disc)
    fname_pkl = fname + '.pkl'
    fpath_pkl = os.path.join('./test', fname_pkl)
    model.write_to_disk(fpath_pkl, int(args.nKeep), int(args.nThin))
    fpath_del = os.path.join('./test', fname + '_delta.csv')
    fpath_gam = os.path.join('./test', fname + '_gamma.pkl')
      
    res = Result(fpath_pkl)
    postdeltas = res.samples.delta
    postalphas = res.generate_conditional_posterior_predictive_gammas()

    pd.DataFrame(postdeltas).to_csv(fpath_del, index = False)
    with open(fpath_gam, 'wb') as file:
        pickle.dump(postalphas, file)

# EOF