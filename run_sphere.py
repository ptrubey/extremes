import sys, os, glob, re
import numpy as np
from subprocess import Popen, PIPE, STDOUT
from argparse import ArgumentParser

source_path = './simulated/sphere/data_*.csv'
dest_path   = './simulated/sphere'
models      = ['sdpppg', 'sdppprg', 'sdpppgln', 'sdppprgln']


def argparser():
    p = ArgumentParser()
    p.add_argument('replace', default = False, action = 'store_true')
    return p.parse_args()

if __name__ == '__main__':
    p = argparser()
    
    files = glob.glob(source_path)
    search_string = r'data_m(\d+)_r(\d+).csv'

    processes = []

    # ensure that only data files are considered
    files = [file for file in files if re.search(search_string, file)]

    for file in files:
        match = re.search(search_string, file)
        
        nMix = match.group(1)
        nCol = match.group(2)

        # outcome  = 'class_m{}.csv'.format(nMix)

        for model in models:
            out_name = 'results_{}_{}_{}.pkl'.format(model, nMix, nCol)
            out_path = os.path.join(dest_path, out_name)
            
            if os.path.exists(out_path) and not p.replace:
                pass
            else:
                processes.append(Popen(
                    [sys.executable, 'test_generic.py', file, out_path, model, '--sphere', 'True']
                    ))
    
    for process in processes:
        process.wait()
    
    rcs = np.array([process.returncode for process in processes])
    print(np.where(rcs != 0))


    

