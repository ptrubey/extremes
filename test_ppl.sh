#!/bin/bash

nohup python postpred_loss.py ./output > /dev/null 2>&1 &
nohup python postpred_loss.py ./output2 > /dev/null 2>&1 &

# EOF
