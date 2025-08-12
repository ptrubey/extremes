"""
test variable bayes algorithms for projected gamma
"""

from projgamma.varbayes import Adam
from projgamma.model_pypprgvb import grad_gamgam_ln, grad_resgamgam_ln
from projgamma.data import Data, euclidean_to_psphere
from numpy.random import gamma, lognormal
import numpy as np


if __name__ == '__main__':
    raw = gamma(
        shape = np.array([5.,3.,2.,0.5]), 
        scale = 1, 
        size = (1000, 4),
        )
    lYs = np.log(raw).sum(axis = 0)
    Ys = raw.sum(axis = 0)
    n  = np.array([raw.shape[0]])
    a = b = np.array([1.])
    t = np.zeros((2,1,4))
    adam = Adam.from_meta(t)
    adam.specify_dloss(
        lambda theta: - grad_resgamgam_ln(theta, lYs, n, a, b),
        )
    for _ in range(50):
        adam.optimize()
    shapes = lognormal(
        mean = np.exp(adam.theta[0,0]), 
        sigma = np.exp(adam.theta[1,0]), 
        size = (500,4),
        )
    print(shapes.mean(axis = 0))

# EOF