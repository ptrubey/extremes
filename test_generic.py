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
    if eval(p.cats):
        assert(p.model.startswith('m'))
    else:
        assert(not p.model.startswith('m'))

    Chain  = models.Chains[p.model]
    Result = models.Results[p.model]
    raw = read_csv(p.in_path).values
    raw = raw[~np.isnan(raw).any(axis = 1)] # equivalent to na.omit

    ## Initialize Data
    if p.model.startswith('c'):
        data = Categorical(raw)
    elif eval(p.cats):
        if eval(p.sphere):
            data = MixedData(raw, eval(p.cats), eval(p.sphere))
        else:
            try:
                data = MixedData(
                    raw, 
                    eval(p.cats), 
                    decluster = eval(p.decluster), 
                    quantile = float(p.quantile),
                    )
            except:
                data = MixedData(
                    raw, 
                    eval(p.cats), 
                    decluster = eval(p.decluster), 
                    quantile = float(p.quantile),
                    )
    else:
        if eval(p.sphere):
            data = Data_From_Sphere(raw)
        else:
            try:
                data = Data_From_Raw(
                    raw, 
                    decluster = eval(p.decluster), 
                    quantile = float(p.quantile),
                    )
            except:
                data = Data_From_Raw(
                    raw, 
                    decluster = eval(p.decluster), 
                    quantile = float(p.quantile),
                    )

    ## If there's a supplied outcome, initialize it
    if os.path.exists(p.outcome):
        outcome = read_csv(p.outcome).values
        data.fill_outcome(outcome)
    
    ## Initialize Chain
    if p.model[:3] in ('sdp','mdp','cdp','mpy'):
        model = Chain(
                    data, 
                    prior_eta = GammaPrior(float(p.eta_shape), float(p.eta_rate)), 
                    p = int(p.p), max_clust_count = int(p.maxclust),
                    ntemps = int(p.ntemps), stepping = float(p.stepping),
                    prior_chi = eval(p.prior_chi),
                    )
    elif p.model[:3] in ('sfm','mfm','cfm'):
        model = Chain(data, nMix = int(p.nMix), p = int(p.p),)
    elif p.model[:2] in ('sv', 'mv', 'cv'):
        model = Chain(data, p = int(p.p),)
    else:
        raise ValueError
    
    ## Run Sampler
    model.sample(int(p.nSamp))

    ## Write to disk
    model.write_to_disk(p.out_path, int(p.nKeep), int(p.nThin))
    
# EOF
