import model_mdln as mdln
import model_mgdln as mgdln
import model_mpgln as mpgln
import model_mprgln as mprgln

import model_dpdln as dpdln
import model_dpgdln as dpgdln
import model_dppgln as dppgln
import model_dpprg as dpprg

Results = {
    'mdln'    : mdln.Result,
    'mgdln'   : mgdln.Result,
    'mpgln'   : mpgln.Result,
    'mprgln'  : mprgln.Result,
    'dpdln'   : dpdln.Result,
    'dpgdln'  : dpgdln.Result,
    'dppgln'  : dppgln.Result,
    'dpprg'   : dpprg.Result,
    }
Chains = {
    'mdln'    : mdln.Chain,
    'mgdln'   : mgdln.Chain,
    'mpgln'   : mpgln.Chain,
    'mprgln'  : mprgln.Chain,
    'dpdln'   : dpdln.Chain,
    'dpgdln'  : dpgdln.Chain,
    'dppgln'  : dppgln.Chain,
    'dpprg'   : dpprg.Chain,
    }

# EOF
