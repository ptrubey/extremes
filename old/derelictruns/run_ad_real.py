import sys, os
import numpy as np
from subprocess import Popen, PIPE, STDOUT

discount = '1e-1'
concentration = '1e-1'
prior_chi = '[{},{}]'.format(discount, concentration)

cardio = {
    'source'    : './ad/cardio/real_data.csv',
    'outcome'   : './ad/cardio/real_outcome.csv',
    'results'   : './ad/cardio/real_results_{}_{}.pkl',
    'quantile'  : '0.85',
    'cats'      : '[15,16,17,18,19,20,21,22,23,24]',
    'decluster' : 'False',
    'model'     : 'pypprgln',    
    }
cover = {
    'source'    : './ad/cover/real_data.csv',
    'outcome'   : './ad/cover/real_outcome.csv',
    'results'   : './ad/cover/real_results_{}_{}.pkl',
    'quantile'  : '0.98',
    'cats'      : '[9,10,11,12]',
    'decluster' : 'False',
    'model'     : 'pypprgln',
    }
mammography = {
    'source'    : './ad/mammography/real_data.csv',
    'outcome'   : './ad/mammography/real_outcome.csv',
    'results'   : './ad/mammography/real_results_{}_{}.pkl',
    'quantile'  : '0.95',
    'cats'      : '[5,6,7,8,9]',
    'decluster' : 'False',
    'model'     : 'pypprgln',
    }
pima = {
    'source'    : './ad/pima/real_data.csv',
    'outcome'   : './ad/pima/real_outcome.csv',
    'results'   : './ad/pima/real_results_{}_{}.pkl',
    'quantile'  : '0.90',
    'cats'      : '[7,8,9,10,11,12]',
    'decluster' : 'False',
    'model'     : 'pypprgln',
    }
annthyroid = {
    'source'    : './ad/annthyroid/real_data.csv',
    'outcome'   : './ad/annthyroid/real_outcome.csv',
    'results'   : './ad/annthyroid/real_results_{}_{}.pkl',
    'quantile'  : '0.85',
    'cats'      : '[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]',
    'decluster' : 'False',
    'model'     : 'pypprgln',
    }
yeast = {
    'source'    : './ad/yeast/real_data.csv',
    'outcome'   : './ad/yeast/real_outcome.csv',
    'results'   : './ad/yeast/real_results_{}_{}.pkl',
    'quantile'  : '0.90',
    'cats'      : '[4,5,8,9,10]',
    'decluster' : 'False',
    'model'     : 'pypprgln',
    }

datasets = [cardio, cover, mammography, annthyroid, yeast, pima]
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
        processes[-1].wait() # operate sequentially rather than in parallel

    # for process in processes:
    #     process.wait()
    
    error_proc_ids = np.where(
        np.array([process.returncode for process in processes]) != 0
        )[0]
    
    for error_proc_id in error_proc_ids:
        print(process_args[error_proc_id])

# EOF
