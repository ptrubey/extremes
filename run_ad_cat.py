import sys, os
import numpy as np
from subprocess import Popen, PIPE, STDOUT

discount = '1e-1'
concentration = '1e-1'
prior_chi = '[{},{}]'.format(discount, concentration)

cover = {
    'source'    : './ad/cover/cat_data.csv',
    'outcome'   : './ad/cover/cat_outcome.csv',
    'results'   : './ad/cover/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7,8,9]',
    }
pima = {
    'source'    : './ad/pima/cat_data.csv',
    'outcome'   : './ad/pima/cat_outcome.csv',
    'results'   : './ad/pima/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7]',
    }
yeast = {
    'source'    : './ad/yeast/cat_data.csv',
    'outcome'   : './ad/yeast/cat_outcome.csv',
    'results'   : './ad/yeast/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7]',
    }
solarflare = {
    'source'    : './ad/solarflare/cat_data.csv',
    'outcome'   : './ad/solarflare/cat_outcome.csv',
    'results'   : './ad/solarflare/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7,8,9]',
    }

bank = {
    'source'    : './ad/bank/cat_data.csv',
    'outcome'   : './ad/bank/cat_outcome.csv',
    'results'   : './ad/bank/cat_results_{}_{}.pkl',
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7,8,9,10,11,12]',
    }
def banki(i):
    bank_i = {
    'source'    : './ad/bank/bank_{}/cat_data.csv'.format(i),
    'outcome'   : './ad/bank/bank_{}/cat_outcome.csv'.format(i),
    'results'   : './ad/bank/bank_{}/cat_results_{}_{}.pkl'.format(i,'{}','{}'),
    'model'     : 'pypprgln',
    'cats'      : '[0,1,2,3,4,5,6,7,8,9,10,11,12]',
    }
    return(bank_i)
bank_is = [banki(i) for i in range(1,11)]

# datasets = [cover, pima, solarflare, yeast]
datasets = [bank] + bank_is

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
        processes[-1].wait() # operate in sequential rather than parallel

    # for process in processes:
    #     process.wait()
    
    error_proc_ids = np.where(
        np.array([process.returncode for process in processes]) != 0
        )[0]
    
    for error_proc_id in error_proc_ids:
        print(process_args[error_proc_id])

# EOF
