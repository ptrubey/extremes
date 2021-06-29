#!/bin/bash

nohup python test_generic.py ./datasets/ad_cardio_x.csv ./ad/cardio dphprg 50000 20000 30 --decluster False > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ad_cover_x.csv ./ad/cover mhprg 50000 20000 30 --nMix 50 --quantile 0.99 --decluster False > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ad_mammography_x.csv ./ad/mammography dphprg 50000 20000 30 --decluster False > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ad_pima_x.csv ./ad/pima dphprg 50000 20000 30 --decluster False > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ad_satellite_x.csv ./ad/satellite dphprg 50000 20000 30 --decluster False > /dev/null 2>&1 &

# EOF
