import pandas as pd
import numpy as np

from projgamma.model_pypprgvb import Chain as VBChain, Result as VBResult
from projgamma.model_pypprg import Chain as MCCHain, Result as MCResult
from projgamma.data import Data

from projgamma.priors import GammaPrior, GEMPrior

from projgamma.model_pypprgvb import grad_resgamgam_ln, grad_gamgam_ln
from projgamma.varbayes import Adam

if __name__ == '__main__':
    
    raise   
    # vbmodel = VBChain(data, p = 10, gibbs_samples = 1000)
    # vbmodel.sample(10000, verbose = True)
    # vbout = vbmodel.to_dict()
    # vbres = VBResult(vbout)
    # vbcond_zetas  = vbres.generate_conditional_posterior_predictive_zetas()
    # vbcond_gammas = vbres.generate_conditional_posterior_predictive_gammas()
    # vbzetas       = vbres.generate_posterior_predictive_zetas()
    # vbgammas      = vbres.generate_posterior_predictive_gammas()

    # mcmodel = MCCHain(data, p = 10)
    # mcmodel.sample(10000, verbose = True)
    # mcout = mcmodel.to_dict()
    # mcres = MCResult(mcout)
    # mccond_zetas  = mcres.generate_conditional_posterior_predictive_zetas()
    # mccond_gammas = mcres.generate_conditional_posterior_predictive_gammas()
    # mczetas       = mcres.generate_posterior_predictive_zetas()
    # mcgammas      = mcres.generate_posterior_predictive_gammas()

# EOF