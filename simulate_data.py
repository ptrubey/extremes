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
        return vnew, alpha, beta, delta

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

        V_df.to_sql('data', conn, index = False)
        a_df.to_sql('alphas', conn, index = False)
        b_df.to_sql('betas', conn, index = False)
        d_df.to_sql('deltas', conn, index = False)
        conn.commit()
        conn.close()
        return

    def __init__(self, nCol, nMix, p, nDat, a0 = 1.8, b0 = 1.2):
        self.nCol, self.nMix, self.nDat = nCol, nMix, nDat
        self.V, self.alpha, self.beta, self.delta = self.simulate_data(nCol, nMix, p, nDat, a0, b0)
        return

class Samples(object):
    alpha = None
    beta  = None
    delta = None

    def __init__(self, nSamp, nCol, nMix):
        self.alpha = np.empty((nSamp, nMix, nCol))
        self.beta  = np.empty((nSamp, nMix, nCol))
        self.delta = np.empty((nSamp, nMix))
        return

class Result(object):
    def load_data(path):
        conn = sql.connect(path)
        alpha = pd.read_sql('select * from alphas;', conn).values
        beta  = pd.read_sql('select * from betas;', conn).values
        delta = pd.read_sql('select * from deltas;', conn).values

        self.samples = Samples(self.nSamp, self.nCol, self.nMix)
        self.samples.alpha[:] = alpha
        self.samples.beta[:] = beta
        self.samples.delta[:] = delta
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(path)
        return

class Data(data.Data):
    V = None

    def read_data(self, path):
        conn = sql.connect(path)
        self.V = pd.read_sql('select * from data;', conn).values
        self.S = data.euclidean_to_simplex(self.V)
        self.Yl = data.angular_to_euclidean(data.euclidean_to_angular(self.V))
        conn.close()
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
