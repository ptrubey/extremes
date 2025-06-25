from anomaly import Anomaly, plot_log_inverse_scores_knn as plot_sorted_scores
from model_mdppprg_pt import Result
import numpy as np

class MixedResult(Result,Anomaly):
    pass

if __name__ == '__main__':
    path = './ad/cardio/results_mdppprg.pkl'
    res = MixedResult(path)

    zetas = np.swapaxes(
        np.array([
            zeta[delta] 
            for zeta, delta 
            in zip(res.samples.zeta, res.samples.delta)
            ]),
        0, 1,
        )
    
    
