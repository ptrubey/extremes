import numpy as np
from scipy.optimize import minimize

def gpd_fit(data, threshold):
    " fits the GPD parameters for a given threshold "
    diff = data - threshold
    excess = diff[diff > 0]
    Nu = excess.shape[0]
    xbar = excess.mean()
    s2   = excess.var()
    xi0  = -0.5 * (((xbar * xbar) / s2) - 1.)
    sc0  = 0.5 * xbar * (((xbar * xbar) / s2) + 1)
    theta = np.array((sc0, xi0))
    def gpd_neg_log_lik(theta):
        sc, xi = theta
        if (sc < 0.) or (xi <= 0 and excess.max() > (- sc / xi)):
            return 1e9
        else:
            y = np.log2(1 + (xi * excess) / sc) / xi
            return len(excess) * np.log2(sc) + (1 + xi) * y.sum()

    fit = minimize(gpd_neg_log_lik, theta, method = 'L-BFGS-B')
    return tuple(fit.x)

# EOF
