import model_sdpppg as sdpppg
import model_sdppprg as sdppprg
import model_sdppprgln as sdppprgln
import model_sdpppgln as sdpppgln
# import model_cdppprg as cdppprg
import model_cdppprgln as cdppprgln
# import model_mdppprg_pt as mdppprg
import model_mdppprgln as mdppprgln
import model_mpypprgln as mpypprgln
import model_pypprgln as pypprgln

## Real Data
RealResults = {
    'sdpppg'    : sdpppg.Result,
    'sdppprg'   : sdppprg.Result,
    'sdpppgln'  : sdpppgln.Result,
    'sdppprgln' : sdppprgln.Result,
    }
RealChains = {
    'sdpppg'    : sdpppg.Chain,
    'sdppprg'   : sdppprg.Chain,
    'sdpppgln'  : sdpppgln.Chain,
    'sdppprgln' : sdppprgln.Chain,
    }

## Mixed Data
MixedResults = {
    # 'mdppprg'   : mdppprg.Result,
    'mdppprgln' : mdppprgln.Result,
    'mpypprgln' : mpypprgln.Result,
    'pypprgln'  : pypprgln.Result,
    }
MixedChains = {
    # 'mdppprg'   : mdppprg.Chain,
    'mdppprgln' : mdppprgln.Chain,
    'mpypprgln' : mpypprgln.Chain,
    'pypprgln'  : pypprgln.Chain,
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
