import silence_tensorflow.auto
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from numpy.random import gamma

from tfprojgamma import ProjectedGamma
np.random.seed(1)
tf.random.set_seed(2)

# alpha_true = gamma(size = (3,5), shape = 1.5)
# pi_true = (0.3, 0.5, 0.2)

# MixProjectedGamma = tfp.distributions.MixtureSameFamily(
#     mixture_distribution = tfd.Categorical(
#         probs = pi_true,
#         ),
#     components_distribution = ProjectedGamma(
#         concentration = alpha_true,
#         )
#     )
# Yp = tf.cast(MixProjectedGamma.sample(1000), tf.float64)

def euclidean_to_psphere(euc, p = 10, EPS = np.finfo(float).eps):
    Yp = (euc + EPS) / (((euc + EPS)**p).sum(axis = -1)**(1/p))[...,None]
    Yp[Yp < EPS] = EPS
    return Yp
Yp = tf.cast(
    euclidean_to_psphere(
        pd.read_csv('~/git/projgamma/datasets/ivt_updated_nov_mar_angular.csv').values,
        ),
    tf.float64,
    )

N, D = Yp.shape; J = 200 # N = nobs, D = ncols, J = nclust
a = 0.5; b = 0.5         # strength (inherently unstable)
c = 2.0; d = 2.0         # rate (biased towards 1)
eta, dis = 0.1, 0.1      # PY Strength / Discount Parameters

# Thanks to Dave Moore for extending this to work with batch dimensions!
# This turns out to be necessary for ADVI to work properly.
def stickbreak(v):
    batch_ndims = len(v.shape) - 1
    cumprod_one_minus_v = tf.math.cumprod(1 - v, axis=-1)
    one_v = tf.pad(
        v, 
        [[0, 0]] * batch_ndims + [[0, 1]], 
        "CONSTANT", 
        constant_values = 1,
        )
    c_one = tf.pad(
        cumprod_one_minus_v, 
        [[0, 0]] * batch_ndims + [[1, 0]], 
        "CONSTANT", 
        constant_values = 1,
        )
    return one_v * c_one

def create_model(N, D, J, eta, discount, dtype = np.float64):
    """
    N : Number of observations
    D : Number of columns
    J : Truncation Point (Max Number of Clusters for Stick-Breaking)
    eta, discount : Pitman Yor Parameters
    """
    model = tfd.JointDistributionNamed(dict(
        xi = tfd.Independent(
            tfd.Gamma(
                concentration = np.full(D, a, dtype),
                rate = np.full(D, b, dtype),
                ),
            reinterpreted_batch_ndims = 1,
            ),
        tau = tfd.Independent(
            tfd.Gamma(
                concentration = np.full(D, c, dtype),
                rate = np.full(D, d, dtype),
                ),
            reinterpreted_batch_ndims = 1,
            ),
        nu = tfd.Independent(
            # tfd.Beta(np.ones(J - 1, dtype) - discount, eta + np.arange(1, J) * discount),
            tfd.Beta(
                tf.ones(J - 1, dtype) - discount, 
                eta + tf.range(1, J, dtype = dtype) * discount,
                ),
            reinterpreted_batch_ndims = 1,
            ),
        alpha = lambda xi, tau: tfd.Independent(
            tfd.Gamma(
                # concentration = np.ones((J, D), dtype) * tf.expand_dims(xi, -2),
                concentration = tf.broadcast_to(xi, (J,D)),
                # rate = np.ones((J, D), dtype) * tf.expand_dims(tau, -2),
                rate = tf.broadcast_to(tau, (J,D))
                ),
            reinterpreted_batch_ndims = 2,
            ),        
        obs = lambda alpha, nu: tfd.Sample(
            tfd.MixtureSameFamily(
                mixture_distribution = tfd.Categorical(probs = stickbreak(nu)),
                components_distribution = ProjectedGamma(alpha, np.ones((J, D), dtype)),
                ),
            sample_shape = (N),
            ),
        ))
    return(model)

model_joint = create_model(N, D, J, eta, dis)
_ = model_joint.sample()
model_joint

def target_log_prob_fn(xi, tau, nu, alpha):
    return model_joint.log_prob(xi = xi, tau = tau, nu = nu, alpha = alpha, obs = Yp)

q_nu_mu    = tf.Variable(tf.random.normal([J-1], dtype = np.float64), name = 'q_nu_mu')
q_nu_sd    = tf.Variable(tf.random.normal([J-1], dtype = np.float64), name = 'q_nu_sd')
q_alpha_mu = tf.Variable(tf.random.normal([J,D], dtype = np.float64), name = 'q_alpha_mu')
q_alpha_sd = tf.Variable(tf.random.normal([J,D], dtype = np.float64), name = 'q_alpha_sd')
q_xi_mu    = tf.Variable(tf.random.normal([D],   dtype = np.float64), name = 'q_xi_mu')
q_xi_sd    = tf.Variable(tf.random.normal([D],   dtype = np.float64), name = 'q_xi_sd')
q_tau_mu   = tf.Variable(tf.random.normal([D],   dtype = np.float64), name = 'q_tau_mu')
q_tau_sd   = tf.Variable(tf.random.normal([D],   dtype = np.float64), name = 'q_tau_sd')

surrogate_posterior = tfd.JointDistributionNamed(dict(
    xi = tfd.Independent(
        tfd.LogNormal(q_xi_mu, tf.nn.softplus(q_xi_sd)), 
        reinterpreted_batch_ndims = 1,
        ),
    tau = tfd.Independent(
        tfd.LogNormal(q_tau_mu, tf.nn.softplus(q_tau_sd)), 
        reinterpreted_batch_ndims = 1,
        ),
    nu = tfd.Independent(
        tfd.LogitNormal(q_nu_mu, tf.nn.softplus(q_nu_sd)), 
        reinterpreted_batch_ndims = 1,
        ),
    alpha = tfd.Independent(
        tfd.LogNormal(q_alpha_mu, tf.nn.softplus(q_alpha_sd)), 
        reinterpreted_batch_ndims = 2,
        ),
    ))

s0 = surrogate_posterior.sample(100)
p = target_log_prob_fn(**s0)

def run_advi(optimizer, sample_size=10, num_steps=2000, seed=2):
    return tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn = target_log_prob_fn,
        surrogate_posterior = surrogate_posterior,
        optimizer = optimizer,
        sample_size = sample_size,
        seed = seed, 
        num_steps = num_steps,
        )

opt = tf.optimizers.Adam(learning_rate=1e-2)
losses = run_advi(opt)

# plt.plot(losses.numpy())
# plt.xlabel('Optimizer Iteration')
# plt.ylabel('ELBO')
# plt.show()

s = surrogate_posterior.sample(1000)
p = stickbreak(s['nu']).numpy()
plt.boxplot(p)
plt.savefig('py_weights.png')

# s1 = surrogate_posterior.sample(1000)
# initlogdens = target_log_prob_fn(**s0).numpy()
# postlogdens = target_log_prob_fn(**s1).numpy()

# plt.plot(initlogdens)
# plt.show()
# plt.plot(postlogdens)
# plt.show()

# plt.boxplot(s0['nu'].numpy())
# plt.show()
# plt.boxplot(s1['nu'].numpy())
# plt.show()


# EOF