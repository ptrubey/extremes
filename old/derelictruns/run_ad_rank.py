import sys, os
import numpy as np
from subprocess import Popen, PIPE, STDOUT

discount = '1e-1'
concentration = '1e-1'
prior_chi = '[{},{}]'.format(discount, concentration)

cardio = {
    'source'    : './ad/cardio/rank_data.csv',
    'outcome'   : './ad/cardio/rank_outcome.csv',
    'results'   : './ad/cardio/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[19,20,21]',
    'model'     : 'pypprgln',
    'model_radius' : 'True',
    }
cover = {
    'source'    : './ad/cover/rank_data.csv',
    'outcome'   : './ad/cover/rank_outcome.csv',
    'results'   : './ad/cover/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[9,10,11,12]',
    'model'     : 'pypprgln',
    'model_radius' : 'True',
    }
mammography = {
    'source'    : './ad/mammography/rank_data.csv',
    'outcome'   : './ad/mammography/rank_outcome.csv',
    'results'   : './ad/mammography/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[6,7,8]',
    'model'     : 'pypprgln',
    'model_radius' : 'True',
    }
annthyroid = {
    'source'    : './ad/annthyroid/rank_data.csv',
    'outcome'   : './ad/annthyroid/rank_outcome.csv',
    'results'   : './ad/annthyroid/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]',
    'model'     : 'pypprgln',
    'model_radius' : 'True',
    }
yeast = {
    'source'    : './ad/yeast/rank_data.csv',
    'outcome'   : './ad/yeast/rank_outcome.csv',
    'results'   : './ad/yeast/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[6,7]',
    'model'     : 'pypprgln',
    'model_radius' : 'True',
    }
pima = {
    'source'    : './ad/pima/rank_data.csv',
    'outcome'   : './ad/pima/rank_outcome.csv',
    'results'   : './ad/pima/rank_results_{}_{}.pkl',
    'realtype'  : 'rank',
    'cats'      : '[8,9,10,11,12]',
    'model'     : 'pypprgln',
    'model_radius' : 'True',
    }
## Categorical

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
            '--realtype', dataset['realtype'],
            '--cats', dataset['cats'],
            '--nSamp', '50000', '--nKeep', '30000', '--nThin', '20',
            '--prior_chi', prior_chi,
            '--ntemps', ntemps,
            '--stepping', stepping,
            '--model_radius', dataset['model_radius'],
            ]
        process_args.append(args)
        processes.append(Popen(args))
        processes[-1].wait() # operate in sequential rather than parallel

    # for process in processes:
    #     process.wait()
    
    error_proc_ids = np.where(
        np.array([process.returncode for process in processes]) != 0
        )[0]
    
    for error_proc_id in error_proc_ids:
        print(process_args[error_proc_id])

# EOF
