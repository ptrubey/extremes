import sys, os
import numpy as np
from subprocess import Popen, PIPE, STDOUT

discount = '1e-1'
concentration = '1e-1'
prior_chi = '[{},{}]'.format(discount, concentration)

cover = {
    'source'    : './ad/cover/cat_data_xv{}_is.csv',
    'outcome'   : './ad/cover/cat_outcome_xv{}_is.csv',
    'results'   : './ad/cover/cat_results_xv{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7,8,9]',
    }
pima = {
    'source'    : './ad/pima/cat_data_xv{}_is.csv',
    'outcome'   : './ad/pima/cat_outcome_xv{}_is.csv',
    'results'   : './ad/pima/cat_results_xv{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7]',
    }
yeast = {
    'source'    : './ad/yeast/cat_data_xv{}_is.csv',
    'outcome'   : './ad/yeast/cat_outcome_xv{}_is.csv',
    'results'   : './ad/yeast/cat_results_xv{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7]',
    }
solarflare = {
    'source'    : './ad/solarflare/cat_data_xv{}_is.csv',
    'outcome'   : './ad/solarflare/cat_outcome_xv{}_is.csv',
    'results'   : './ad/solarflare/cat_results_xv{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7,8,9]',
    }

datasets = [cover, pima, solarflare, yeast]

stepping = '1.1'
ntemps = '5'

if __name__ == '__main__':
    processes = []
    process_args = []

    for dataset in datasets:
        for xv_ in range(5):
            xv = xv_ + 1
            args = [
                sys.executable,
                'test_generic.py', 
                dataset['source'].format(xv),
                dataset['results'].format(xv),
                dataset['model'],
                '--outcome', dataset['outcome'].format(xv),
                '--realtype', 'rank',
                '--cats', dataset['cats'],
                '--nSamp', '50000', '--nKeep', '30000', '--nThin', '40',
                '--prior_chi', prior_chi,
                '--ntemps', ntemps,
                '--stepping', stepping,
                '--model_radius', 'False',
                ]
            process_args.append(args)
            processes.append(Popen(args))

    for process in processes:
        process.wait()
    
    error_proc_ids = np.where(
        np.array([process.returncode for process in processes]) != 0
        )[0]
    
    for error_proc_id in error_proc_ids:
        print(process_args[error_proc_id])

# EOF
