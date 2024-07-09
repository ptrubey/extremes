import numpy as  np
import pandas as pd
import os

from data import RealData


path_full = './datasets/slosh/filtered_data.csv.gz'
path_t90  = './datasets/slosh/slosh_thr.90.csv.gz'

if __name__ == '__main__':
    slosh = pd.read_csv(path_full, compression = 'gzip')
    slosh_t90 = pd.read_csv(path_t90, compression='gzip').values
    ltd = ~slosh.MTFCC.isin(['C3061','C3081'])
    apt = slosh.MTFCC.isin(['K2451'])
    slosh_ltd = slosh.T[8:].T[ltd].values.T.astype(float)
    slosh_apt = slosh.T[8:].T[apt].values.T.astype(float)
    slosh_t90 = slosh_t90.T[8:].astype(float)
    data_ltd = RealData(slosh_ltd, 'threshold', False, 0.95)
    data_apt = RealData(slosh_apt, 'threshold', False, 0.95)
    data_t90 = RealData(slosh_t90, 'threshold', False, 0.90)

    datas = [data_ltd, data_apt, data_t90]
    Ns = [data.nDat for data in datas]
    Ss = [data.nCol for data in datas]
    names = ['ltd','apt','t90']
    Is = []
    for data in datas:
        I = np.zeros(4000, dtype = int)
        I[data.I] = 1
        Is.append(I)
    
    Is = np.array(Is).T
    desc = pd.DataFrame({'data' : names, 'N' : Ns, 'S' : Ss})
    desc.to_csv('./test/threshold_summary.csv', index = False)
    pd.DataFrame(Is).to_csv('./test/threshold_inclusion.csv', index = False)

# EOF
    