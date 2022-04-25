import pandas as pd, numpy as np
import os, glob, re
from energy import limit_cpu
from anomaly import ResultFactory

base_path = './simulated/sphere_ad'
model_types = np.array(['sdpppg','sdppprg','sdpppgln', 'sdppprgln'])

if __name__ == '__main__':
    limit_cpu()
    paths = glob.glob(os.path.join(base_path, 'results_*.pkl'))
    metrics = []
    for path in paths:
        match = re.search('results_([a-zA-Z]+)_(\d+)_(\d+).pkl', path)
        model, nmix, ncol = match.group(1,2,3)
        print('Processing {} m{} c{}'.format(model, nmix, ncol).ljust(76), end = '')
        result = ResultFactory(model, path)
        result.pools_open()
        metric = result.get_scoring_metrics()
        result.pools_closed()
        metric['Model'] = model
        metric['nMix']  = nmix
        metric['nCol']  = ncol
        column_order = ['Model','nMix','nCol','Metric'] + list(result.scoring_metrics.keys()) 
        metrics.append(metric[column_order])
        print('Done')
    
    df = pd.concat(metrics)
    df.to_csv(os.path.join(base_path, 'performance.csv'), index = False)

# EOF
