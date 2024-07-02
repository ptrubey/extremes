"""
Run the SLOSH regression model for different datasets, at different values of
concentration and discount
"""

import sys, os, glob, re
import numpy as np
from subprocess import Popen, PIPE, STDOUT

# srces = ['ltd','apt']
srces = ['ltd']
concs = ['0.01','0.1','0.5','2']
discs = ['0.001','0.01','0.05','0.1']

if __name__ == '__main__':
    processes = []
    for srce in srces:
        for conc in concs:
            for disc in discs:
                processes.append(Popen(
                    [sys.executable, 'test_reg.py', srce, conc, disc,]
                    ))
    
    for process in processes:
        process.wait()
    
    rcs = np.array([process.returncode for process in processes])
    print(np.where(rcs != 0))

# EOF