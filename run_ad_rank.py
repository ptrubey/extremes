import sys, os
import numpy as np
from subprocess import Popen, PIPE, STDOUT

discount = '1e-1'
concentration = '1e0'
prior_chi = '[{},{}]'.format(discount, concentration)

cardio = {
    'source'    : './ad/cardio/data_new.csv',
    'outcome'   : './ad/cardio/outcome_new.csv',
    'results'   : './ad/cardio/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[19,20,21]',
    'model'     : 'mpypprgln',    
    }
cover = {
    'source'    : './ad/cover/data_new.csv',
    'outcome'   : './ad/cover/outcome_new.csv',
    'results'   : './ad/cover/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[9,10,11,12]',
    'model'     : 'mpypprgln',
    }
mammography = {
    'source'    : './ad/mammography/data_new.csv',
    'outcome'   : './ad/mammography/outcome_new.csv',
    'results'   : './ad/mammography/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[6,7,8]',
    'model'     : 'mpypprgln',
    }
annthyroid = {
    'source'    : './ad/annthyroid/data_new.csv',
    'outcome'   : './ad/annthyroid/outcome_new.csv',
    'results'   : './ad/annthyroid/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]',
    'model'     : 'mpypprgln',
    }
yeast = {
    'source'    : './ad/yeast/data_new.csv',
    'outcome'   : './ad/yeast/outcome_new.csv',
    'results'   : './ad/yeast/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[6,7]',
    'model'     : 'mpypprgln',
    }
pima = {
    'source'    : './ad/pima/data_new.csv',
    'outcome'   : './ad/pima/outcome_new.csv',
    'results'   : './ad/pima/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[8,9,10,11,12]',
    'model'     : 'mpypprgln',
    }
## Categorical

datasets = [cardio, cover, mammography, annthyroid, yeast, pima]
# datasets = [mammography, annthyroid, yeast]
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
            '--realtype', dataset['realtype'],
            '--cats', dataset['cats'],
            '--nSamp', '50000', '--nKeep', '30000', '--nThin', '20',
            '--prior_chi', prior_chi,
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
