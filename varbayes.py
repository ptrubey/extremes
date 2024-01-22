import numpy as np
import projgamma as pg
import data
from scipy.special import digamma
from samplers import bincount2D_vectorized

def gradient_normal(x, mu, sigma):
    """
    calculates gradient on 
    """


def gradient_pypg_alpha(alpha, xi, tau, delta, log_y, logs_y):
    """
    calculates gradient on alpha_{jl} (shape parameter) for sample of size S
    on projected gamma distribution with product of gammas prior
    alpha   : PG shape              (S, J, D)
    xi      : PG prior shape        (S, D)
    tau     : PG prior rate         (S, D)
    delta   : cluster identifier    (S, N)
    log_y   : log(y)                (N, D)
    logs_y  : log(sum(y))           (N)
    """
    S, J, D = alpha.shape
    assert (D == xi.shape[1]) and (D == tau.shape[1]) and (D == log_y.shape[1])
    assert (S == xi.shape[0]) and (S == tau.shape[0]) and (S == delta.shape[0])
    assert (delta.shape[1] == log_y.shape[0])
    assert (log_y.shape[0] == logs_y.shape[0])

    dmat = delta[:,:,None] == range(J)     # (S, N, J)
    n_j  = bincount2D_vectorized(delta, J) # (S, J)
    s_a  = alpha.sum(axis = -1)            # (S, J)
    
    out = np.zeros((S, J, D))
    out += (n_j * digamma(s_a))[:,:,None]  # (S, J, 1)
    out -= np.einsum('snj,n->sj', dmat, logs_y)[:,:,None] # (S, J, 1)
    out += np.einsum('snj,nd->sjd', dmat, log_y)  # (S, J, D)
    out += (n_j[:,:,None] + xi[:,None,:] - 1) * digamma(alpha) # (S,J,D)
    return out

def gradient_pypg_xi(alpha, xi, tau, a, b):
    """
    calculates gradient on xi_{l} (prior shape parameter) for sample of size S
    for Projected Gamma distribution with product of gammas prior
    alpha : PG Shape            (S, J, D)
    xi    : PG Prior Shape      (S, D)
    tau   : PG Prior Rate       (S, D)
    a     : xi Prior Shape      (1)
    b     : xi Prior Rate       (1)
    """
    S, J, D = alpha.shape
    assert (S == xi.shape[0]) and (S == tau.shape[0])
    assert (D == xi.shape[1]) and (D == tau.shape[1])

    out = np.zeros((S, D))
    out += J * (np.log(tau) - digamma(xi))
    out += np.log(alpha).sum(axis = 1)
    out += (a - 1) / xi
    out -= b
    return out

def gradient_pypg_tau(alpha, xi, tau, c, d):
    """
    calculates gradient on tau_{l} for sample of size S
    alpha : PG Shape            (S, J, D)
    xi    : PG Prior Shape      (S, D)
    tau   : PG Prior Rate       (S, D)
    a     : xi Prior Shape      (1)
    b     : xi Prior Rate       (1)
    """
    S, J, D = alpha.shape

    out = np.zeros((S, D))
    out += (J * xi + c - 1) / tau
    out -= alpha.sum(axis = 1)
    out -= d
    return out

def stickbreak(nu):
    """
        Stickbreaking cluster probability
        nu : (S x (J - 1))
    """
    lognu = np.log(nu)
    log1mnu = np.log(1 - nu)

    S = nu.shape[0]; J = nu.shape[1] + 1
    out = np.zeros((S,J))
    out[:,:-1] + np.log(nu)
    out[:, 1:] += np.cumsum(np.log(1 - nu))
    return np.exp(out)

class VarPYPG:
    """ 
        Variational Approximation of Pitman-Yor Mixture of Projected Gammas 
    """
    def __init__(self, data, max_clusters):
        self.data = data
        pass

    pass

class MVarPYPG:
    """ 
        Variational Approximation of Pitman-Yor Mixture of Projected Gammas
        with exact sampling of cluster membership / cluster weights
    """
    def __init__(self, data, max_clusters):
        self.data = data
        self.max_clusters = max_clusters
        pass
    
    pass

# EOF