import numpy as np
import pandas as pd
import sqlite3 as sql
import os
import data
from numpy.random import gamma, choice

class Chain(object):
    def simulate_data(self, nCol, nMix, p, nSamp, a0 = 1.8, b0 = 1.2):
        alpha = gamma(a0, scale = 1/b0, size = (nMix, nCol))
        beta  = gamma(a0, scale = 1/b0, size = (nMix, nCol))
        p /= p.sum()
        delta = choice(nMix, size = nSamp, p = p)

        anew = alpha[delta]
        bnew = beta[delta]

        gnew = gamma(anew, scale = 1/bnew, size = (nSamp, nCol))
        vnew = data.euclidean_to_hypercube(gnew)
        return vnew, alpha, beta, delta, p

    def write_to_disk(self, path):
        if not os.path.exists(os.path.split(path)[0]):
            os.mkdir(os.path.split(path)[0])
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        V_df = pd.DataFrame(self.V, columns = ['V_{}'.format(i) for i in range(self.nCol)])
        a_df = pd.DataFrame(self.alpha, columns = ['alpha_{}'.format(i) for i in range(self.nCol)])
        b_df = pd.DataFrame(self.beta, columns = ['beta_{}'.format(i) for i in range(self.nCol)])
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
        self.V, self.alpha, self.beta, self.delta, self.p = self.simulate_data(nCol, nMix, p, nDat, a0, b0)
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
        conn    = sql.connect(path)
        self.V  = pd.read_sql('select * from data;', conn).values
        self.S  = data.euclidean_to_simplex(self.V)
        self.Yl = data.angular_to_euclidean(data.euclidean_to_angular(self.V))
        self.A  = data.euclidean_to_angular(self.Yl)
        conn.close()
        return

    def __init__(self, path):
        self.read_data(path)
        self.nDat, self.nCol = self.V.shape
        return

if __name__ == '__main__':
    nCols = [3,6,9,12]
    nMixs = [3,6,9,12]

    for nCol in nCols:
        for nMix in nMixs:
            chain = Chain(nCol = nCol, nMix = nMix, p = np.ones(nMix) / nMix, nDat = 500)
            print('Col {} Mix {}'.format(nCol, nMix))
            chain.write_to_disk('./simulated/sim_c{}_m{}/data.db'.format(nCol, nMix))
            pass
        pass
# EOF
