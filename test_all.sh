#!/bin/bash
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output/new/ivt9_result_sdpppg.pkl sdpppg > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output/new/ivt9_result_sdppprg.pkl sdppprg > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output/new/ivt9_result_sdppprgln.pkl sdppprgln > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_nov_mar.csv ./output/new/ivt9_result_sdpppgln.pkl sdpppgln  > /dev/null 2>&1 &

nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output/new/ivt43_result_sdpppg.pkl sdpppg > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output/new/ivt43_result_sdppprg.pkl sdppprg > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output/new/ivt43_result_sdpppgln.pkl sdpppgln > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output/new/ivt43_result_sdppprgln.pkl sdppprgln > /dev/null 2>&1 &

# EOF
