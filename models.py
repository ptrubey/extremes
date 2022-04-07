import model_sdpppg as sdpppg
import model_sdppprg as sdppprg
import model_sdppprgln as sdppprgln
import model_sdpppgln as sdpppgln
import model_mdppprg as mdppprg
import model_mdpppg as mdpppg

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
    'mdppprg'  : mdppprg.Result,
    'mdpppg'   : mdpppg.Result,
    }
MixedChains = {
    'mdppprg'  : mdppprg.Chain,
    'mdpppg'   : mdpppg.Chain,
    }

Results = {**MixedResults, **RealResults}
Chains  = {**MixedChains,  **RealChains}

# EOF
