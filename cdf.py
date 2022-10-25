"""
cdf.py

Functions for computing (and storing) empirical CDF's, 
    computing rank transformations, etc.
"""
import numpy as np
EPS = np.finfo(float).eps

class ECDF(object):
    X = None
    N = None

    def Fhat(self, Xnew):
        return np.searchsorted(self.X, Xnew, side = 'right') / self.N

    def stdpareto(self, Xnew):
        return (1 / (1 - self.Fhat(Xnew) + EPS)) + EPS

    def __init__(self, X):
        assert len(X.shape) == 1 # verify univariate input
        self.X
        self.X = np.sort(X)
        self.N = X.shape[0]
        return

if __name__ == '__main__':
    from numpy.random import uniform
    X = uniform(size = 1000000)
    F = ECDF(X)
    print(F.stdpareto(np.array([0.0000001, 0.5, 0.99999999])))

# EOF