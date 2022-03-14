import sys, os, glob, re
import numpy as np
from subprocess import Popen, PIPE, STDOUT

source_path = './datasets/sim_mixed/*.csv'
dest_path   = './sim_mixed_ad'
models      = ['mdpppg'] #, 'mdppprg']

if __name__ == '__main__':
    files = glob.glob(source_path)
    search_string = r'data_m(\d+)_r(\d+)_c(\d+).csv'

    processes = []

    files = [file for file in files if re.search(search_string, file)]

    for file in files:
        match = re.search(search_string, file)
        
        nMix = match.group(1)
        nCol = match.group(2)
        nCat = match.group(3)

        outcome  = 'class_m{}.csv'.format(nMix)

        for model in models:
            out_name = 'results_{}_{}_{}_{}.pkl'.format(model, nMix, nCol, nCat)
            log_name = 'log_{}_{}_{}_{}.log'.format(model, nMix, nCol, nCat)
            out_path = os.path.join(dest_path, out_name)
            
            catCols = list(np.arange(int(nCol) + int(nCat))[int(nCol):])
            
            processes.append(Popen(
                [sys.executable, 'test_generic.py', file, out_path, model, 
                    '--outcome', out_path, "--cats", str(catCols), '--sphere', 'True']
                ))
    
    for process in processes:
        process.wait()
    
    rcs = np.array([process.returncode for process in processes])
    print(np.where(rcs != 0))


    

