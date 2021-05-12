import numpy as np
import psutil
import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
from sklearn.metrics import pairwise_distances
from hypercube_deviance import hcdev

def hypercube_distance(args):
    return pairwise_distances(args[0], args[1].reshape(1,-1), metric = hcdev).sum()

def prediction_pairwise_distance(prediction):
    n = prediction.shape[0]
    res = map(hypercube_distance, zip(repeat(prediction), prediction))
    return np.array(list(res)).sum() / (n * n)

def target_pairwise_distance(args):
    prediction, target = args
    n = prediction.shape[0]
    return hypercube_distance((prediction, target)) / n

def limit_cpu():
    p = psutil.Process(os.getpid())
    if os.name == 'nt':
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif os.name == 'posix':
        p.nice(10)
    else:
        pass
    return

def energy_score(predictions, targets):
    pool = Pool(processes = cpu_count(), initializer = limit_cpu)
    res1 = pool.map(prediction_pairwise_distance, predictions)
    res2 = pool.map(target_pairwise_distance, zip(predictions, targets))
    pool.close()
    return np.array(list(res2)).mean() - 0.5 * np.array(list(res1)).mean()

def intrinsic_energy_score(dataset):
    res1 = prediction_pairwise_distance(dataset) # same for all elements of df.  only do once.
    pool = Pool(processes = cpu_count())
    res2 = pool.map(target_pairwise_distance, zip(repeat(dataset),dataset))
    pool.close()
    return np.array(list(res2)).mean() - 0.5 * res1

if __name__ == '__main__':
    pass

# EOF
