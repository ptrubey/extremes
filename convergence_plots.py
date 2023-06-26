import numpy as np
import pandas as pd
import pickle as pkl
import glob
import os
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsp

def title_maker(path, addendum = ''):
    pathbase = '{};  {}-{}{}'
    dataset = 'ERA-Interim' * ('ivt9' in path) + 'ERA-5' * ('ivt43' in path)
    model = 'PG' * ('sdpppg' in path) + 'PRG' * ('sdppprg' in path)
    prior = 'LN' * ('ln' in path) + 'G' * ('ln' not in path)
    return pathbase.format(dataset, model, prior, addendum)

def path_maker(file_path, out_base, addendum = ''):
    pathbase = '{}-{}-{}{}.pdf'
    dataset = (
            'ERA-Interim' * ('ivt9' in file_path) 
        +   'ERA-5' * ('ivt43' in file_path)
        )
    model = 'PG' * ('sdpppg' in file_path) + 'PRG' * ('sdppprg' in file_path)
    prior = 'LN' * ('ln' in file_path) + 'G' * ('ln' not in file_path)
    out = os.path.join(
        out_base, pathbase.format(dataset, model, prior, addendum)
        )
    return out

def plot_convergence(path, nburn, out_base):
    """ Plots Log-density over time
    inputs:
        path: path to Result pickle
        nburn: Number of burn samples
        out_base: path to output folder
    """
    with open(path, 'rb') as file:
        res = pkl.load(file)
    logd = res['logd'][1:]
    s = np.arange(logd.shape[0])
    fig, ax = plt.subplots()
    ax.plot(s, logd)
    ax.set(xlabel='Iteration',ylabel='Log Density',title=title_maker(path))
    ax.vlines(
        nburn, 
        ymin = logd.min() - 100, 
        ymax = logd.max() + 100, 
        colors = 'red'
        )
    fig.savefig(path_maker(path, out_base))
    plt.clf()
    return

def plot_acf(path, nburn, out_base):
    """ Plots AutoCorrelation function (after nburn)
    inputs:
        path: path to Result pickle
        nburn: Number of burn samples (calculate acf after this)
        out_base: path to output folder
    """
    with open(path, 'rb') as file:
        res = pkl.load(file)
    logd = res['logd'][1:,]
    fig, ax = plt.subplots()
    tsp.plot_acf(logd[nburn:], ax = ax, lags = 100)
    ax.set(title = title_maker(path, '  Autocorrelation'))
    fig.savefig(path_maker(path, out_base, '-acf'))    
    plt.clf()
    pass

if __name__ == '__main__':
    res_path = './output/new/ivt*'
    out_path = '../pgpareto/images'
    paths = glob.glob(res_path)
    for path in paths:
        plot_convergence(path, 40000, out_path)
        plot_acf(path, 40000, out_path)
    pass


# EOF 