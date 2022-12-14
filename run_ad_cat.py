import sys, os
import numpy as np
from subprocess import Popen, PIPE, STDOUT

discount = '1e-1'
concentration = '1e-1'
prior_chi = '[{},{}]'.format(discount, concentration)

cardio = {
    'source'    : './ad/cardio/data_cat.csv',
    'outcome'   : './ad/cardio/outcome_cat.csv',
    'results'   : './ad/cardio/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',    
    }
cover = {
    'source'    : './ad/cover/data_cat.csv',
    'outcome'   : './ad/cover/outcome_cat.csv',
    'results'   : './ad/cover/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7,8,9]',
    }
mammography = {
    'source'    : './ad/mammography/data_cat.csv',
    'outcome'   : './ad/mammography/outcome_cat.csv',
    'results'   : './ad/mammography/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    }
pima = {
    'source'    : './ad/pima/data_cat.csv',
    'outcome'   : './ad/pima/outcome_cat.csv',
    'results'   : './ad/pima/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7]',
    }
satellite = {
    'source'    : './ad/satellite/data_cat.csv',
    'outcome'   : './ad/satellite/outcome_cat.csv',
    'results'   : './ad/satellite/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    }
annthyroid = {
    'source'    : './ad/annthyroid/data_cat.csv',
    'outcome'   : './ad/annthyroid/outcome_cat.csv',
    'results'   : './ad/annthyroid/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    }
yeast = {
    'source'    : './ad/yeast/data_cat.csv',
    'outcome'   : './ad/yeast/outcome_cat.csv',
    'results'   : './ad/yeast/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7]',
    }
solarflare = {
    'source'    : './ad/solarflare/data_cat.csv',
    'outcome'   : './ad/solarflare/outcome_cat.csv',
    'results'   : './ad/solarflare/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7,8,9]',
    }


# datasets = [cardio, cover, mammography, annthyroid, yeast, pima]
datasets = [cover, pima, solarflare, yeast]

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
            '--realtype', 'rank',
            '--cats', dataset['cats'],
            '--nSamp', '50000', '--nKeep', '30000', '--nThin', '20',
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
