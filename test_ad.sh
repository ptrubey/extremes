#!/bin/bash

# nohup python test_generic.py ./datasets/ad_cardio_x.csv ./ad/cardio dphprg 50000 20000 30 --eta_rate 1e0 --decluster False > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ad_cover_x.csv ./ad/cover mhprg 50000 20000 30 --nMix 50 --decluster False > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ad_mammography_x.csv ./ad/mammography dphprg 50000 20000 30 --eta_rate 1e0 --decluster False > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ad_pima_x.csv ./ad/pima dphprg 50000 20000 30 --eta_rate 1e0 --decluster False > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ad_satellite_x.csv ./ad/satellite dphprg 50000 20000 30 --eta_rate 1e0 --decluster False --quantile 0.97 > /dev/null 2>&1 &

nohup python test_generic_ad.py ./simulated_ad/ad_sim_m5_c5_x.csv ./simulated_ad/m5_c5 dphprg 50000 20000 30 --decluster False --quantile 0.> /dev/null 2>&1 &
nohup python test_generic_ad.py ./simulated_ad/ad_sim_m5_c10_x.csv ./simulated_ad/m5_c10 dphprg 50000 20000 30 --decluster False --quantile 0.> /dev/null 2>&1 &
nohup python test_generic_ad.py ./simulated_ad/ad_sim_m10_c5_x.csv ./simulated_ad/m10_c5 dphprg 50000 20000 30 --decluster False --quantile 0.> /dev/null 2>&1 &
nohup python test_generic_ad.py ./simulated_ad/ad_sim_m10_c10_x.csv ./simulated_ad/m10_c10 dphprg 50000 20000 30 --decluster False --quantile 0.> /dev/null 2>&1 &

# EOF
