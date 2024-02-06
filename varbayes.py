import numpy as np
import pandas as pd
import projgamma as pg
from scipy.special import digamma
from samplers import bincount2D_vectorized
from data import Data
import matplotlib.pyplot as plt
import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from tfprojgamma import ProjectedGamma

def gradient_normal(x):
    """
    calculates gradient on no
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

def stickbreak_tf(nu):
    batch_ndims = len(nu.shape) - 1
    cumprod_one_minus_nu = tf.math.cumprod(1 - nu, axis=-1)
    one_v = tf.pad(nu, [[0, 0]] * batch_ndims + [[0, 1]], "CONSTANT", constant_values=1)
    c_one = tf.pad(cumprod_one_minus_nu, [[0, 0]] * batch_ndims + [[1, 0]], "CONSTANT", constant_values=1)
    return one_v * c_one

class MVarPYPG:
    """ 
        Variational Approximation of Pitman-Yor Mixture of Projected Gammas
        with exact sampling of cluster membership / cluster weights
    """
    def __init__(self, data, max_clusters = 200):
        self.data = data
        self.max_clusters = max_clusters
        pass
    
    pass

class SurrogateVars(object):
    def init_vars(self, J, D, dtype):
        self.q_nu_mu    = tf.Variable(
            tf.random.normal([J-1], dtype = dtype), name = 'q_nu_mu',
            )
        self.q_nu_sd    = tf.Variable(
            tf.random.normal([J-1], dtype = dtype), name = 'q_nu_sd',
            )
        self.q_alpha_mu = tf.Variable(
            tf.random.normal([J,D], dtype = dtype), name = 'q_alpha_mu',
            )
        self.q_alpha_sd = tf.Variable(
            tf.random.normal([J,D], dtype = dtype), name = 'q_alpha_sd',
            )
        self.q_xi_mu    = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'q_xi_mu',
            )
        self.q_xi_sd    = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'q_xi_sd',
            )
        self.q_tau_mu   = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'q_tau_mu',
            )
        self.q_tau_sd   = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'q_tau_sd',
            )
        return
        
    def __init__(self, J, D, dtype = np.float64):
        self.init_vars(J, D, dtype)
        return

class SurrogateModel(object):
    def init_vars(self):
        self.vars = SurrogateVars(self.J, self.D, self.dtype)
        return
    
    def init_model(self):
        self.model = tfd.JointDistributionNamed(dict(
            xi = tfd.Independent(
                tfd.LogNormal(
                    self.vars.q_xi_mu, tf.nn.softplus(self.vars.q_xi_sd),
                    ), 
                reinterpreted_batch_ndims = 1,
                ),
            tau = tfd.Independent(
                tfd.LogNormal(
                    self.vars.q_tau_mu, tf.nn.softplus(self.vars.q_tau_sd),
                    ), 
                reinterpreted_batch_ndims = 1,
                ),
            nu = tfd.Independent(
                tfd.LogitNormal(
                    self.vars.q_nu_mu, tf.nn.softplus(self.vars.q_nu_sd),
                    ), 
                reinterpreted_batch_ndims = 1,
                ),
            alpha = tfd.Independent(
                tfd.LogNormal(
                    self.vars.q_alpha_mu, tf.nn.softplus(self.vars.q_alpha_sd),
                    ), 
                reinterpreted_batch_ndims = 2,
                ),
            ))
        return

    def __init__(self, J, D, dtype = np.float64):
        self.J = J
        self.D = D
        self.dtype = dtype
        self.init_vars()
        self.init_model()
        return

class VarPYPG(object):
    """ 
        Variational Approximation of Pitman-Yor Mixture of Projected Gammas
        Constructed using TensorFlow, AutoDiff
    """
    start_time, end_time, time_elapsed = None, None, None

    def init_model(self):
        self.model = tfd.JointDistributionNamed(dict(
            xi = tfd.Independent(
                tfd.Gamma(
                concentration = np.full(self.D, self.a, self.dtype),
                rate = np.full(self.D, self.b, self.dtype),
                ),
                reinterpreted_batch_ndims = 1,
                ),
            tau = tfd.Independent(
                tfd.Gamma(
                    concentration = np.full(self.D, self.c, self.dtype),
                    rate = np.full(self.D, self.d, self.dtype),
                    ),
                reinterpreted_batch_ndims = 1,
                ),
            nu = tfd.Independent(
                tfd.Beta(
                    np.ones(self.J - 1, self.dtype) - self.discount, 
                    self.eta + np.arange(1, self.J) * self.discount
                    ),
                reinterpreted_batch_ndims = 1,
                ),
            alpha = lambda xi, tau: tfd.Independent(
                tfd.Gamma(
                    concentration = np.ones(
                        (self.J, self.D), self.dtype,
                        ) * tf.expand_dims(xi, -2),
                    rate = np.ones(
                        (self.J, self.D), self.dtype,
                        ) * tf.expand_dims(tau, -2),
                    ),
                reinterpreted_batch_ndims = 2,
                ),        
            obs = lambda alpha, nu: tfd.Sample(
                tfd.MixtureSameFamily(
                    mixture_distribution = tfd.Categorical(probs = stickbreak_tf(nu)),
                    components_distribution = ProjectedGamma(
                        alpha, np.ones((self.J, self.D), self.dtype)
                        ),
                    ),
                sample_shape = (self.N),
                ),
            ))
        _ = self.model.sample()
        def log_prob_fn(xi, tau, nu, alpha):
            return self.model.log_prob(
                xi = xi, tau = tau, nu = nu, alpha = alpha, obs = self.data.Yp,
                )
        self.log_prob_fn = log_prob_fn
        return

    def init_surrogate(self):
        self.surrogate = SurrogateModel(self.J, self.D, self.dtype)
        return
    
    def fit_advi(self, num_steps = 5000, sample_size = 50, seed = 1):
        optimizer = tf.optimizers.Adam(learning_rate=1e-2)
        self.start_time = time.time()
        losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn = self.log_prob_fn,
            surrogate_posterior = self.surrogate.model,
            optimizer = optimizer,
            sample_size = sample_size,
            seed = seed,
            num_steps = num_steps,
            )
        self.end_time = time.time()
        self.time_elapsed = self.end_time - self.start_time
        return(losses)

    def __init__(
            self, 
            data, 
            eta = 0.1, 
            discount = 0.1, 
            prior_xi = (0.5, 0.5), 
            prior_tau = (2., 2.), 
            max_clusters = 200,
            dtype = np.float64
            ):
        self.data = data
        self.J = max_clusters
        self.N = self.data.nDat
        self.D = self.data.nCol
        self.a, self.b = prior_xi
        self.c, self.d = prior_tau
        self.eta = eta
        self.discount = discount
        self.dtype = dtype
        self.init_model()
        self.init_surrogate()
        return

    pass

if __name__ == '__main__':
    np.random.seed(1)
    tf.random.set_seed(1)

    slosh = pd.read_csv(
        './datasets/slosh/filtered_data.csv.gz', 
        compression = 'gzip',
        )
    slosh_ids = slosh.T[:8].T
    slosh_obs = slosh.T[8:].T
    
    sloshes = dict()
    for category in slosh_ids.Category.unique():
        idx = (slosh_ids.Category == category)
        ids = slosh_ids[idx]
        obs = slosh_obs[idx].values.T.astype(np.float64)
        dat = Data(obs, real_vars = np.arange(obs.shape[1]), quantile = 0.95)
        mod = VarPYPG(dat)
        mod.fit_advi()
        raise

        sloshes[category] = [ids.shape[0], mod.time_elapsed]

    for slosh in sloshes.keys():
        print(sloshes[slosh])


raise
# EOF