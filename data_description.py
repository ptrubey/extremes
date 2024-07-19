import numpy as  np
import pandas as pd
import os
import glob

from data import RealData
from collections import namedtuple

def data_description():
    path_apt  = './datasets/slosh/slosh_apt_data.csv.gz'
    path_ltd  = './datasets/slosh/slosh_ltd_data.csv.gz'
    path_t90  = './datasets/slosh/slosh_t90_data.csv.gz'
    path_emg  = './datasets/slosh/slosh_emg_data.csv.gz'
    path_xpt  = './datasets/slosh/slosh_xpt_data.csv.gz'

    slosh_apt = pd.read_csv(
        path_apt, compression = 'gzip',
        ).T[8:].values.astype(float)
    slosh_ltd = pd.read_csv(
        path_ltd, compression = 'gzip',
        ).T[8:].values.astype(float)
    slosh_t90 = pd.read_csv(
        path_t90, compression = 'gzip',
        ).T[8:].values.astype(float)
    slosh_emg = pd.read_csv(
        path_emg, compression = 'gzip',
        ).T[8:].values.astype(float)
    slosh_xpt = pd.read_csv(
        path_xpt, compression = 'gzip',
        ).T[8:].values.astype(float)
    data_ltd = RealData(slosh_ltd, 'threshold', False, 0.95)
    data_apt = RealData(slosh_apt, 'threshold', False, 0.95)
    data_t90 = RealData(slosh_t90, 'threshold', False, 0.90)
    data_emg = RealData(slosh_emg, 'threshold', False, 0.95)
    data_xpt = RealData(slosh_xpt, 'threshold', False, 0.95)

    datas = [data_ltd, data_apt, data_t90, data_emg, data_xpt]
    Ns = [data.nDat for data in datas]
    Ss = [data.nCol for data in datas]
    names = ['ltd','apt','t90', 'emg', 'xpt']
    Is = []
    for data in datas:
        I = np.zeros(4000, dtype = int)
        I[data.I] = 1
        Is.append(I)
    
    Is = np.array(Is).T
    desc = pd.DataFrame({'data' : names, 'N' : Ns, 'S' : Ss})
    desc.to_csv('./test/threshold_summary.csv', index = False)
    pd.DataFrame(Is).to_csv('./test/threshold_inclusion.csv', index = False)

def extant_clusters():
    path_wildcard = './test/sloshltd_*_delta.csv'
    ParmSet = namedtuple('ParmSet', 'dataset conc disc ncluster')

    def extract_parms(path):
        fnames = os.path.splitext(os.path.split(path)[1])[0].split('_')
        data = fnames[1]
        conc = fnames[2]
        disc = fnames[3]
        df = pd.read_csv(path)
    
        def nunique(series):
            return np.unique(series).shape[0]
    
        n = np.array([nunique(df.iloc[i]) for i in range(df.shape[0])])
    
        return ParmSet(data, conc, disc, n.mean())

    paths = glob.glob(path_wildcard)
    parms = []
    for path in paths:
        try:
            parms.append(extract_parms(path))
        except pd.errors.EmptyDataError:
            print('Failed {}'.format(path))
    df = pd.DataFrame(parms)
    df.to_csv('./test/sloshltd_cluster_counts.csv', index = False)

if __name__ == '__main__':
    data_description()
    extant_clusters()

# EOF
