"""
test variable bayes algorithms for projected gamma
"""

from projgamma.varbayes import Adam
from projgamma.model_pypprgvb import grad_gamgam_ln, grad_resgamgam_ln
from projgamma.data import Data, euclidean_to_psphere
from numpy.random import gamma, lognormal, normal
import numpy as np


if __name__ == '__main__':
    # raw = pd.read_csv('./datasets/ivt_nov_mar.csv').values
    # data = Data.from_raw(raw, xh1t_cols = np.arange(raw.shape[1]), dcls = True, xhquant = 0.95)
    Y = np.vstack((
        gamma(shape = np.array([2.0, 0.2, 0.3, 5.0]), size = (300, 4)),
        gamma(shape = np.array([0.2, 3.0, 0.1, 5.0]), size = (200, 4)),
        gamma(shape = np.array([6.0, 0.3, 2.0, 3.0]), size = (100, 4)),
        gamma(shape = np.array([0.3, 0.5, 3.0, 0.5]), size = (150, 4)),
        ))
    delta = np.array(
        [0] * 300 + [1] * 200 + [2] * 100 + [3] * 150, 
        dtype = int,
        )
    alpha = np.array([1.] * 4)
    beta  = np.array([1.] * 4)
    
    dmat  = delta[:,None] == np.arange(20)
    lYs = dmat.T @ np.log(Y)
    Ys  = dmat.T @ Y
    n   = dmat.sum(axis = 0)

    thet = normal(size = [2, *lYs.shape])
    adam = Adam.from_meta(thet)
    l = lambda theta: - grad_resgamgam_ln(theta, lYs, n, alpha, beta)
    adam.specify_dloss(l)
    for _ in range(1000):
        if _ % 100 == 0:
            print(thet[0,:4])
        adam.optimize()

# EOF