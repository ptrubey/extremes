import sys, os
import numpy as np
from subprocess import Popen, PIPE, STDOUT

discount = '1e-1'
concentration = '1e-1'
prior_chi = '[{},{}]'.format(discount, concentration)

cardio = {
    'source'    : './ad/cardio/data.csv',
    'outcome'   : './ad/cardio/outcome.csv',
    'results'   : './ad/cardio/results_{}_{}.pkl',
    'quantile'  : '0.85',
    'cats'      : '[15,16,17,18,19,20,21,22,23,24]',
    'decluster' : 'False',
    'model'     : 'mpypprgln',    
    }
cover = {
    'source'    : './ad/cover/data.csv',
    'outcome'   : './ad/cover/outcome.csv',
    'results'   : './ad/cover/results_{}_{}.pkl',
    'quantile'  : '0.98',
    'cats'      : '[9,10,11,12]',
    'decluster' : 'False',
    'model'     : 'mpypprgln',
    }
mammography = {
    'source'    : './ad/mammography/data.csv',
    'outcome'   : './ad/mammography/outcome.csv',
    'results'   : './ad/mammography/results_{}_{}.pkl',
    'quantile'  : '0.95',
    'cats'      : '[5,6,7,8,9]',
    'decluster' : 'False',
    'model'     : 'mpypprgln',
    }
pima = {
    'source'    : './ad/pima/data.csv',
    'outcome'   : './ad/pima/outcome.csv',
    'results'   : './ad/pima/results_{}_{}.pkl',
    'quantile'  : '0.90',
    'cats'      : '[7,8,9,10,11,12]',
    'decluster' : 'False',
    'model'     : 'mpypprgln',
    }
satellite = {
    'source'    : './ad/satellite/data.csv',
    'outcome'   : './ad/satellite/outcome.csv',
    'results'   : './ad/satellite/results_{}_{}.pkl',
    'quantile'  : '0.95',
    'cats'      : '[36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]',
    'decluster' : 'False',
    'model'     : 'mpypprgln',
    }
annthyroid = {
    'source'    : './ad/annthyroid/data.csv',
    'outcome'   : './ad/annthyroid/outcome.csv',
    'results'   : './ad/annthyroid/results_{}_{}.pkl',
    'quantile'  : '0.85',
    'cats'      : '[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]',
    'decluster' : 'False',
    'model'     : 'mpypprgln',
    }
yeast = {
    'source'    : './ad/yeast/data.csv',
    'outcome'   : './ad/yeast/outcome.csv',
    'results'   : './ad/yeast/results_{}_{}.pkl',
    'quantile'  : '0.90',
    'cats'      : '[4,5,8,9,10]',
    'decluster' : 'False',
    'model'     : 'mpypprgln',
    }
## Categorical
solarflare = {
    'source'    : './ad/solarflare/data.csv',
    'outcome'   : './ad/solarflare/outcome.csv',
    'results'   : './ad/solarflare/results_{}_{}.pkl',
    'quantile'  : 'None',
    'cats'      : 'None',
    'decluster' : 'False',
    'model'     : 'cdppprgln',
    }


# datasets = [cardio, cover, mammography, annthyroid, yeast, pima]
# datasets = [mammography, annthyroid, yeast]
datasets = [cardio,cover,pima]
stepping = '1.1'
ntemps = '5'

if __name__ == '__main__':
    processes = []
    process_args = []

    for dataset in datasets:
        args = [
            sys.executable, 
            'test_generic.py', 
            dataset['source'],
            dataset['results'].format(discount, concentration),
            dataset['model'],
            '--outcome', dataset['outcome'],
            '--cats', dataset['cats'],
            '--quantile', dataset['quantile'],
            '--nSamp', '50000', '--nKeep', '30000', '--nThin', '20',
            '--prior_chi', prior_chi,
            '--decluster', dataset['decluster'],
            '--ntemps', ntemps,
            '--stepping', stepping,
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
