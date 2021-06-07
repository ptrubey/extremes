import model_vd as vd
import model_vgd as vgd
import model_vpg as vpg
import model_vprg as vprg

import model_md as md
import model_mgd as mgd
import model_mpg as mpg
import model_mprg as mprg

import model_dpd as dpd
import model_dpgd as dpgd
import model_dppg as dppg
import model_dpprg as dpprg

import model_dppn as dppn

# testing
import model_dpdc as dpdc
import model_dphpg as dphpg

Results = {
    'vd'    : vd.Result,
    'vgd'   : vgd.Result,
    'vpg'   : vpg.Result,
    'vprg'  : vprg.Result,
    'md'    : md.Result,
    'mgd'   : mgd.Result,
    'mpg'   : mpg.Result,
    'mprg'  : mprg.Result,
    'dpd'   : dpd.Result,
    'dpgd'  : dpgd.Result,
    'dppg'  : dppg.Result,
    'dpprg' : dpprg.Result,
    'dppn'  : dppn.Result,
    'dpdc'  : dpdc.Result,
    'dphpg' : dphpb.Result,
    }
Chains = {
    'vd'    : vd.Chain,
    'vgd'   : vgd.Chain,
    'vpg'   : vpg.Chain,
    'vprg'  : vprg.Chain,
    'md'    : md.Chain,
    'mgd'   : mgd.Chain,
    'mpg'   : mpg.Chain,
    'mprg'  : mprg.Chain,
    'dpd'   : dpd.Chain,
    'dpgd'  : dpgd.Chain,
    'dppg'  : dppg.Chain,
    'dpprg' : dpprg.Chain,
    'dppn'  : dppn.Chain,
    'dpdc'  : dpdc.Chain,
    'dphpg' : dphpg.Chain,
    }

# EOF
