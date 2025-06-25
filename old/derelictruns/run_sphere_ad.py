import sys, os, glob, re
import numpy as np
from subprocess import Popen, PIPE, STDOUT

source_path = './simulated/sphere_ad/data_*.csv'
dest_path   = './simulated/sphere_ad'
models      = ['sdpppg', 'sdppprg']

if __name__ == '__main__':
    files = glob.glob(source_path)
    search_string = r'data_m(\d+)_r(\d+).csv'

    processes = []

    # ensure that only data files are considered
    files = [file for file in files if re.search(search_string, file)]

    for file in files:
        match = re.search(search_string, file)
        
        nMix = match.group(1)
        nCol = match.group(2)

        outcome  = 'class_m{}.csv'.format(nMix)

        for model in models:
            out_name = 'results_{}_{}_{}.pkl'.format(model, nMix, nCol)
            out_path = os.path.join(dest_path, out_name)
            cla_path = os.path.join(os.path.split(file)[0], outcome)
            
            processes.append(Popen(
                [sys.executable, 'test_generic.py', file, out_path, model, 
                '--outcome', cla_path, '--sphere', 'True']
                ))
    
    for process in processes:
        process.wait()
    
    rcs = np.array([process.returncode for process in processes])
    print(np.where(rcs != 0))


    

