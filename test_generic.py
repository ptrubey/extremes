from argparser import argparser_generic as argparser
from data import Data_From_Raw, Data_From_Sphere, MixedData, Categorical
from projgamma import GammaPrior
from pandas import read_csv
from energy import limit_cpu
import models
import os
import numpy as np

if __name__ == '__main__':
    p = argparser()
    limit_cpu()
    # Verify using a mixed model if using categorical variables
    # -- or not using a mixed model, if not using categorical variables
    Chain  = models.Chains[p.model]
    Result = models.Results[p.model]
    raw = read_csv(p.in_path).values
    raw = raw[~np.isnan(raw).any(axis = 1)] # equivalent to na.omit

    ## Initialize Data
    if eval(p.cats):
        data = MixedData(
            raw, cat_vars = eval(p.cats), realtype = p.realtype, 
            decluster = eval(p.decluster), quantile = float(p.quantile),
            )
    else:
        data = Data_From_Raw(
            raw, decluster = eval(p.decluster), quantile=float(p.quantile),
            )
    if os.path.exists(p.outcome):
        raw_out = read_csv(p.outcome).values
        raw_out = raw_out[~np.isnan(raw).any(axis = 1)].ravel().astype(int)
        assert raw_out.shape[0] == raw.shape[0]
        data.fill_outcome(raw_out)
    
    ## Initialize Chain
    model = Chain(
        data, 
        prior_eta = GammaPrior(*eval(p.prior_eta)),
        prior_chi = eval(p.prior_chi),
        p = int(p.p),
        max_clust_count = int(p.maxclust),
        ntemps = int(p.ntemps), 
        stepping = float(p.stepping),
        nMix = int(p.nMix),
        model_radius = eval(p.model_radius),
        )
        
    ## Run Sampler
    model.sample(int(p.nSamp), verbose = True)

    ## Write to disk
    model.write_to_disk(p.out_path, int(p.nKeep), int(p.nThin))
    
# EOF
