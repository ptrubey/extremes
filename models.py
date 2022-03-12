import remodel_dpppg as dpppg
import remodel_dppprg as dppprg
import remodel_dppprgln as dppprgln
import remodel_dpppgln as dpppgln
import mixed_dppprg as mdppprg
import mixed_dpppg as mdpppg

## Real Data
RealResults = {
    'dpppg'    : dpppg.Result,
    'dppprg'   : dppprg.Result,
    'dpppgln'  : dpppgln.Result,
    'dppprgln' : dppprgln.Result,
    }
RealChains = {
    'dpppg'    : dpppg.Chain,
    'dppprg'   : dppprg.Chain,
    'dpppgln'  : dpppgln.Chain,
    'dppprgln' : dppprgln.Chain,
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
