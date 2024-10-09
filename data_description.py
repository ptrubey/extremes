import numpy as  np
import pandas as pd
import os
import glob

from data import RealData
from collections import namedtuple, defaultdict

def data_description():
    """
    Writes data description for each dataset. 

    Number of columns, number of rows that survive thresholding.
    Also outputs dataset of row inclusion (survived thresholding).
    """
    qtl = defaultdict(lambda: 0.90)
    # qtl['t90'] = 0.9
    # dss = ['apt', 'ltd', 't90', 'emg', 'xpt', 'loc', 'del', 'nyc']
    dss = ['t90','dbg','del','crt']

    paths = {
        ds : './datasets/slosh/slosh_{}_data.csv.gz'.format(ds) for ds in dss
        }
    sloss = {
        ds : pd.read_csv(
            paths[ds], compression = 'gzip',
            ).T[8:].values.astype(float) 
        for ds in dss
        }
    datas = {ds : RealData(sloss[ds], 'threshold', False, qtl[ds]) for ds in dss}
    Nobsv = {ds : datas[ds].nDat for ds in dss}
    Nlocs = {ds : datas[ds].nCol for ds in dss}
    Is    = {}
    for ds in dss:
        I = np.zeros(4000, dtype = int)
        I[datas[ds].I] = 1
        Is[ds] = I
    Is = pd.DataFrame(Is)
    desc = pd.DataFrame({
        'data' : dss,
        'N'    : [Nobsv[ds] for ds in dss],
        'S'    : [Nlocs[ds] for ds in dss],
        })    
    desc.to_csv('./test/threshold_summary.csv', index = False)
    Is.to_csv('./test/threshold_inclusion.csv', index = False)
    return

# def extant_clusters():
#     """ Number of extant clusters per dataset """
#     path_wildcard = './test/sloshltd_*_delta.csv'
#     ParmSet = namedtuple('ParmSet', 'dataset conc disc ncluster')

#     def extract_parms(path):
#         fnames = os.path.splitext(os.path.split(path)[1])[0].split('_')
#         data = fnames[1]
#         conc = fnames[2]
#         disc = fnames[3]
#         df = pd.read_csv(path)
    
#         def nunique(series):
#             return np.unique(series).shape[0]
    
#         n = np.array([nunique(df.iloc[i]) for i in range(df.shape[0])])
    
#         return ParmSet(data, conc, disc, n.mean())

#     paths = glob.glob(path_wildcard)
#     parms = []
#     for path in paths:
#         try:
#             parms.append(extract_parms(path))
#         except pd.errors.EmptyDataError:
#             print('Failed {}'.format(path))
#     df = pd.DataFrame(parms)
#     df.to_csv('./test/sloshltd_cluster_counts.csv', index = False)

def extant_clusters():
    """ number of extant clusters per dataset """
    datasets = ['crt','del','dbg','t90']
    path_vb_wildcard = './test/ncluster_vb_*.csv'
    path_mc_wildcard = './test/ncluster_mc_*.csv'
    path_vbs = glob.glob(path_vb_wildcard)
    path_mcs = glob.glob(path_mc_wildcard)
    out = {}
    for path in path_vbs + path_mcs:
        if any([ds in path for ds in datasets]):
            out[path] = pd.read_csv(path)
    lines = []
    Line = namedtuple('Line', 'path clusters')
    for path in out:
        temp = out[path]
        nc = temp.loc[(temp.eta == 0.1) & (temp.discount == 0.1)].ncluster[0]
        lines.append(Line(path, nc))        

    return out

if __name__ == '__main__':
    # data_description()
    extant_clusters()

# EOF
