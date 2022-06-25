""" online updating of covariance matrices """

import numpy as np

class OnlineCovariance(object):
    nCol = None
    A = None
    b = None
    x = None
    n = None 
    Sigma = None

    def update(self, x):
        """ 
        x : (d)
        """
        self.A += x * x[:,None]
        self.b += x
        self.n += 1
        self.xbar[:] = self.b / self.n
        self.Sigma[:] = (1/self.n) * (
            + self.A 
            - self.xbar * self.b[:,None]
            - self.b * self.xbar[:,None]
            + self.n * self.xbar * self.xbar[:,None]
            )
        return

    def __init__(self, nCol):
        self.nCol = nCol
        self.Sigma = np.empty((nCol, nCol))
        self.A = np.zeros((nCol, nCol))
        self.b = np.zeros((nCol))
        self.xbar = np.zeros((nCol))
        self.n = 0
        return

class TemperedOnlineCovariance(OnlineCovariance):
    def update(self, x):
        """" x : (t, d) """
        self.A += np.einsum('tj,tl->tjl', x, x)
        self.b += x
        self.n += 1 
        self.xbar[:] = self.b / self.n
        self.Sigma[:] = 0.
        self.Sigma += self.A
        self.Sigma -= np.einsum('tj,tl->tjl', self.xbar, self.b)
        self.Sigma -= np.einsum('tj,tl->tjl', self.b, self.xbar)
        self.Sigma += np.einsum('tj,tl->tjl', self.xbar, self.xbar)
        self.Sigma /= self.n
        return
    
    def __init__(self, nTemp, nCol):
        self.nTemp, self.nCol = nTemp, nCol
        self.Sigma = np.empty((nTemp, nCol, nCol))
        self.A = np.zeros((nTemp, nCol, nCol))
        self.b = np.zeros((nTemp, nCol))
        self.xbar = np.zeros((nTemp, nCol))
        self.n = 0
        return
    
class PerObsTemperedOnlineCovariance(OnlineCovariance):
    c_Sigma = None
    c_xbar  = None
    c_n     = None

    def cluster_covariance(self, delta):
        """ 
        Combines Covariance Matrices for all elements in cluster 
        adapted from: https://tinyurl.com/onlinecovariance
        """
        # cluster related
        S = np.zeros((self.nTemp, self.nClust, self.nCol, self.nCol))
        mS = np.zeros((self.nTemp, self.nClust, self.nCol))
        nS = np.zeros((self.nTemp, self.nClust))
        # combined (temporary) values
        mC = np.zeros((self.nTemp, self.nCol))
        nC = np.zeros((self.nTemp))
        for j in range(self.nDat):
            nC[:] = nS[self.temps, delta.T[j]] + self.n
            mC[:] = 0.
            mC += nS[self.temps, delta.T[j]] * mS[self.temps, delta.T[j]]
            mC += self.n * self.xbar[j]
            mC /= nC[:,None]
            S[self.temps, delta.T[j]] *= nS[self.temps, delta.T[j], None, None]
            S[self.temps, delta.T[j]] += self.n * self.Sigma[j]
            S[self.temps, delta.T[j]] += np.einsum(
                't,tp,tq->tpq',
                nS[self.temps, delta.T[j]],
                mS[self.temps, delta.T[j]] - mC,
                mS[self.temps, delta.T[j]] - mC,
                )
            S[self.temps, delta.T[j]] += np.einsum(
                'tp,tq->tpq', self.xbar[j] - mC, self.xbar[j] - mC,
                )
            S[self.temps, delta.T[j]] /= nC[:, None, None]
            mS[self.temps, delta.T[j]] = mC
            nS[self.temps, delta.T[j]] = nC
        S += np.eye(self.nCol)[None,None,:] * 1e-9
        return S

    def update(self, x):
        """ x : (n, t, d) """
        self.A += np.einsum('ntj,ntl->ntjl', x, x)
        self.b += x
        self.n += 1
        self.xbar[:] = self.b / self.n
        self.Sigma[:] = 0.
        self.Sigma += self.A
        self.Sigma -= np.einsum('ntj,ntl->ntjl', self.xbar, self.b)
        self.Sigma -= np.einsum('ntj,ntl->ntjl', self.b, self.xbar)
        self.Sigma += np.einsum('ntj,ntl->ntjl', self.xbar, self.xbar)
        self.Sigma /= self.n
    
    def __init__(self, nTemp, nDat, nCol, nClust = None):
        # regular
        self.nTemp, self.nDat, self.nCol = nTemp, nDat, nCol
        self.temps = np.arange(self.nTemp)
        self.Sigma = np.empty((self.nDat, self.nTemp, self.nCol, self.nCol))
        self.A = np.zeros((self.nDat, self.nTemp, self.nCol, self.nCol))
        self.b = np.zeros((self.nDat, self.nTemp, self.nCol))
        self.xbar = np.zeros((self.nDat, self.nTemp, self.nCol))
        self.n = 0
        # clustering
        if nClust is not None:
            self.nClust  = nClust
            self.c_Sigma = np.zeros((self.nTemp, self.nClust, self.nCol, self.nCol))
            self.c_xbar  = np.zeros((self.nTemp, self.nClust, self.nCol))
            self.c_n     = np.zeros((self.nTemp, self.nClust))
        return

if __name__ == "__main__":
    from numpy.linalg import cholesky
    from numpy.random import normal

    starting_cov = np.array([3,0.3,-1.3,0.3,2,0.7,-1.3,0.7,2.5]).reshape(3,3)
    starting_cho = cholesky(starting_cov)

    z = normal(size = (1000, 3))
    x = (starting_cho @ z.T).T
    ccovs = np.empty((1000, 3, 3))
    covs = np.empty((1000, 3, 3))
    means = np.empty((1000, 3))
    cmeans = np.empty((1000, 3))
    covs[0] = 0.
    means[0] = 0.

    sigma = OnlineCovariance(3)

    for i in range(1000):
        sigma.update(x[i])
        covs[i] = sigma.Sigma
        if i > 1:
            ccovs[i] = np.cov(x[:(i+1)].T) * i/(i+1)

# EOF







