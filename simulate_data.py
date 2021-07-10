import numpy as np
import pandas as pd
import sqlite3 as sql
import os
import data
from numpy.random import gamma, choice, uniform, pareto

class Chain(object):
    def simulate_data(self, nCol, nMix, p, nSamp, a0 = 4., b0 = 2.):
        alpha = uniform(0.1, a0, size = (nMix, nCol)) + gamma(1, size = (nMix, nCol))
        beta  = uniform(0.2, b0, size = (nMix, nCol)) + gamma(1, size = (nMix, nCol))
        p /= p.sum()
        delta = choice(nMix, size = nSamp, p = p)

        anew = alpha[delta]
        bnew = beta[delta]

        gnew = gamma(anew, scale = 1/bnew, size = (nSamp, nCol))
        # vnew = data.euclidean_to_hypercube(gnew)
        return gnew, alpha, beta, delta, p

    def write_to_disk(self, path, cols):
        if not os.path.exists(os.path.split(path)[0]):
            os.mkdir(os.path.split(path)[0])
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)

        V_df = pd.DataFrame(
                data.euclidean_to_hypercube(self.G.T[:cols].T),
                columns = ['V_{}'.format(i) for i in range(cols)],
                )
        a_df = pd.DataFrame(self.alpha.T[:cols].T, columns = ['alpha_{}'.format(i) for i in range(cols)])
        b_df = pd.DataFrame(self.beta.T[:cols].T, columns = ['beta_{}'.format(i) for i in range(cols)])
        d_df = pd.DataFrame(self.delta.reshape(1,-1), columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        p_df = pd.DataFrame(self.p.reshape(1,-1), columns = ['p_{}'.format(i) for i in range(self.nMix)])

        V_df.to_sql('data', conn, index = False)
        a_df.to_sql('alphas', conn, index = False)
        b_df.to_sql('betas', conn, index = False)
        d_df.to_sql('deltas', conn, index = False)
        p_df.to_sql('ps', conn, index = False)

        conn.commit()
        conn.close()
        return

    def __init__(self, nCol, nMix, p, nDat, a0 = 1.8, b0 = 1.2):
        self.nCol, self.nMix, self.nDat = nCol, nMix, nDat
        self.G, self.alpha, self.beta, self.delta, self.p = self.simulate_data(nCol, nMix, p, nDat, a0, b0)
        return

class ChainAD(object):
    def simulate_data(self, p, pa, a0 = 4., b0 = 2.):
        alpha = uniform(0.1, a0, size = (self.nMix, self.nCol)) + gamma(1, size = (self.nMix, self.nCol))
        beta  = uniform(0.2, b0, size = (self.nMix, self.nCol)) + gamma(1, size = (self.nMix, self.nCol))

        a_alpha = np.roll(alpha, 1, axis = 1)
        a_beta  = np.roll(beta, 1, axis = 1)

        delta = choice(self.nMix, size = int(self.nSamp * (1 - pa)), p = p)
        a_delta = choice(self.nMix, size = int(self.nSamp * pa), p = p)

        gnew = gamma(alpha[delta], scale = 1 / beta[delta], size = (delta.shape[0], self.nCol))
        a_gnew = gamma(a_alpha[a_delta], scale = 1 / a_beta[a_delta], size = (a_delta.shape[0], self.nCol))

        Gnew = np.vstack((gnew, a_gnew))
        Ynew = np.array([0] * gnew.shape[0] + [1] * a_gnew.shape[0], dtype = int)

        o = choice(self.nSamp, self.nSamp, replace = False)
        return Gnew[o], Ynew[o]

    def write_to_disk(self, path, nCol):
        if not os.path.exists(path):
            os.mkdir(path)

        V = data.euclidean_to_hypercube(self.G.T[:nCol].T)
        R = pareto(1, size = self.nSamp) * (np.ones(self.nSamp) + 0.3 * self.y)

        Z = (V.T * R).T

        Z_df = pd.DataFrame(Z, columns = ['Z_{}'.format(i) for i in range(nCol)])
        y_df = pd.DataFrame({'y' : self.y})

        z_path = os.path.join(path, 'ad_sim_m{}_c{}_x.csv'.format(self.nMix, nCol))
        y_path = os.path.join(path, 'ad_sim_m{}_c{}_y.csv'.format(self.nMix, nCol))

        Z_df.to_csv(z_path, index = False)
        y_df.to_csv(y_path, index = False)
        return

    def __init__(self, nCol, nMix, p, pa, nDat, a0 = 1.8, b0 = 1.2):
        self.nCol, self.nMix, self.nSamp = nCol, nMix, nDat
        self.G, self.y = self.simulate_data(p, pa, a0, b0)
        return

class Samples(object):
    alpha = None
    beta  = None
    delta = None
    p     = None

    def __init__(self, nSamp, nDat, nCol, nMix):
        self.alpha = np.empty((nSamp, nMix, nCol))
        self.beta  = np.empty((nSamp, nMix, nCol))
        self.delta = np.empty((nSamp, nDat), dtype = int)
        self.p     = np.empty((nSamp, nMix))
        return

class Result(object):
    def load_data(self, path):
        conn = sql.connect(path)
        alpha = pd.read_sql('select * from alphas;', conn).values
        beta  = pd.read_sql('select * from betas;', conn).values
        delta = pd.read_sql('select * from deltas;', conn).values.reshape(-1).astype(int)
        p     = pd.read_sql('select * from ps;', conn).values.reshape(-1)

        self.nMix = p.shape[0]

        self.samples = Samples(self.nSamp, self.nDat, self.nCol, self.nMix)
        self.samples.alpha[:] = alpha
        self.samples.beta[:] = beta
        self.samples.delta[:] = delta
        self.samples.p[:] = p
        return

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1):
        postpred = np.empty((self.nSamp, n_per_sample, self.nCol))
        for n in range(self.nSamp):
            delta_new = choice(self.nMix, n_per_sample, p = self.samples.p[n])
            alpha_new = self.samples.alpha[n][delta_new]
            beta_new  = self.samples.beta[n][delta_new]
            postpred[n] = euclidean_to_hypercube(
                gamma(shape = alpha_new, scale = 1/beta_new, size = (n_per_sample, self.nCol))
                )
        return postpred.reshape(-1, self.nCol)

    def __init__(self, path):
        self.data = Data(path)
        self.nCol, self.nDat = self.data.nCol, self.data.nDat
        self.load_data(path)
        return

class Data(data.Data):
    V  = None
    S  = None
    Yl = None

    def read_data(self, path):
        conn     = sql.connect(path)
        self.V   = pd.read_sql('select * from data;', conn).values
        self.S   = data.euclidean_to_simplex(self.V)
        self.Yl  = data.angular_to_euclidean(data.euclidean_to_angular(self.V))
        self.A   = data.euclidean_to_angular(self.Yl)
        self.Vi  = self.cast_to_cube(self.A)
        self.pVi = self.probit(self.Vi)
        conn.close()
        return

    def __init__(self, path):
        self.read_data(path)
        self.nDat, self.nCol = self.V.shape
        return

class DataAD(data.Data):
    def read_data(self, path):
        self.Z = pd.read_csv(path).values
        self.R = self.Z.max(axis = 1)
        self.V = (self.Z.T / self.R).T
        self.S = data.euclidean_to_simplex(self.V)
        self.A = data.euclidean_to_angular(self.V)
        self.Yl = data.angular_to_euclidean(self.A)
        self.Vi = self.cast_to_cube(self.A)
        self.pVi = self.probit(self.Vi)
        self.I = (np.ones(self.V.shape[0], dtype = bool),)
        return

    def __init__(self, path):
        self.read_data(path)
        self.nDat, self.nCol = self.V.shape
        return

if __name__ == '__main__':
    # nCols = [3, 6, 12, 20]
    # nMixs = [3, 6,  9, 12]
    #
    # for nMix in nMixs:
    #     chain = Chain(nCol = max(nCols), nMix = nMix, p = np.ones(nMix) / nMix, nDat = 500)
    #     for nCol in nCols:
    #         print('mix {} col {}'.format(nMix, nCol))
    #         chain.write_to_disk('./simulated/sim_c{}_m{}/data.db'.format(nCol, nMix), nCol)
    #         pass
    #     pass

    nCols = [5,10]
    nMixs = [5,10]

    for nMix in nMixs:
        chain = ChainAD(max(nCols), nMix, np.ones(nMix) / nMix, 0.05, 800)
        for nCol in nCols:
            chain.write_to_disk('./simulated_ad', nCol)
    pass



# EOF
