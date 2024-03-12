import model_sdpppg    as sdpppg
import model_sdppprg   as sdppprg
import model_sdppprgln as sdppprgln
import model_sdpppgln  as sdpppgln
# import model_cdppprg as cdppprg
import model_cdppprgln as cdppprgln
# import model_mdppprg_pt as mdppprg
import model_mdppprgln as mdppprgln
import model_mpypprgln as mpypprgln
import model_pypprgln  as pypprgln
import model_fpypprgln as fpypprgln
import model_gpypprgln as gpypprgln
import model_sd        as sd
import model_spypg     as spypg
import model_spypgiiln as spypgiiln

## Real Data
RealResults = {
    'sdpppg'    : sdpppg.Result,
    'sdppprg'   : sdppprg.Result,
    'sdpppgln'  : sdpppgln.Result,
    'sdppprgln' : sdppprgln.Result,
    'sd'        : sd.Result,
    'spypg'     : spypg.Result,
    'spypgiiln' : spypgiiln.Result,
    }
RealChains = {
    'sdpppg'    : sdpppg.Chain,
    'sdppprg'   : sdppprg.Chain,
    'sdpppgln'  : sdpppgln.Chain,
    'sdppprgln' : sdppprgln.Chain,
    'sd'        : sd.Chain,
    'spypg'     : spypg.Chain,
    'spypgiiln' : spypgiiln.Chain,
    }

## Mixed Data
MixedResults = {
    # 'mdppprg'   : mdppprg.Result,
    'mdppprgln' : mdppprgln.Result,
    'mpypprgln' : mpypprgln.Result,
    'pypprgln'  : pypprgln.Result,
    'fpypprgln'  : fpypprgln.Result,
    'gpypprgln'  : gpypprgln.Result,
    }
MixedChains = {
    # 'mdppprg'   : mdppprg.Chain,
    'mdppprgln' : mdppprgln.Chain,
    'mpypprgln' : mpypprgln.Chain,
    'pypprgln'  : pypprgln.Chain,
    'fpypprgln'  : fpypprgln.Chain,
    'gpypprgln'  : gpypprgln.Chain,
    }

## Categorical Data
CategoricalResults = {
    # 'cdppprg'   : cdppprg.Result,
    'cdppprgln' : cdppprgln.Result,
    }
CategoricalChains = {
    # 'cdppprg'  : cdppprg.Chain,
    'cdppprgln'  : cdppprgln.Chain,
    }


Results = {**MixedResults, **RealResults, **CategoricalResults}
Chains  = {**MixedChains,  **RealChains,  **CategoricalChains}

# EOF
