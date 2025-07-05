import pandas as pd
import numpy as np

from projgamma.model_pypprgvb import Chain as VBChain, Result as VBResult
from projgamma.model_pypprg import Chain as MCCHain, Result as MCResult
from projgamma.data import Data

if __name__ == '__main__':
    raw = pd.read_csv('./datasets/ivt_nov_mar.csv').values
    data = Data.from_raw(raw, xh1t_cols = np.arange(raw.shape[1]), dcls = True, xhquant = 0.95)
    
    # vbmodel = VBChain(data, p = 10, gibbs_samples = 1000)
    # vbmodel.sample(5000, verbose = True)
    # vbout = vbmodel.to_dict()
    # vbres = VBResult(vbout)
    # vbcond_zetas  = vbres.generate_conditional_posterior_predictive_zetas()
    # vbcond_gammas = vbres.generate_conditional_posterior_predictive_gammas()
    # vbzetas       = vbres.generate_posterior_predictive_zetas()
    # vbgammas      = vbres.generate_posterior_predictive_gammas()

    mcmodel = MCCHain(data, p = 10)
    mcmodel.sample(10000, verbose = True)
    mcout = mcmodel.to_dict()
    mcres = MCResult(mcout)
    mccond_zetas  = mcres.generate_conditional_posterior_predictive_zetas()
    mccond_gammas = mcres.generate_conditional_posterior_predictive_gammas()
    mczetas       = mcres.generate_posterior_predictive_zetas()
    mcgammas      = mcres.generate_posterior_predictive_gammas()

# EOF