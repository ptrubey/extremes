#!/bin/bash

nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output dpd 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output dpgd 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output dppg 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output dpprg 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &

nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output md 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output mgd 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output mpg 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output mprg 50000 20000 30 --nMix 30 > /dev/null 2>&1 &

nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output vd 50000 20000 30 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output vgd 50000 20000 30 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output vpg 50000 20000 30 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output vprg 50000 20000 30 > /dev/null 2>&1 &

nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output dppn 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &

# EOF
