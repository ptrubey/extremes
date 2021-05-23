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
    pool = Pool(processes = cpu_count(), initializer = limit_cpu)
    res2 = pool.map(target_pairwise_distance, zip(repeat(dataset),dataset))
    pool.close()
    return np.array(list(res2)).mean() - 0.5 * res1

def knn_distance(X, k, metric):
    distance = pairwise_distances(X, metric = metric)
    knn_dist = np.empty((X.shape[0], k))
    for i in range(X.shape[0]):
        knn_dist[i] = np.sort(distance[i])[1 : k + 1] # distance to itself is always 0...
    return knn_dist

def knnx_distance(X, Y, k, metric):
    xdistance = pairwise_distances(X, Y, metric = metric)
    knnx_dist = np.empty((X.shape[0], k))
    for i in range(X.shape[0]):
        knnx_dist[i] = np.sort(xdistance[i])[:k]
    return knnx_dist

def knn_kl_divergence(empirical, postpred, k = 10, metric = hcdev):
    """ Implements FNN::KL.divergence in python -- Some remaining issues.  Numbers don't match
        for Euclidean pairwise Distances. """
    d1 = knn_distance(empirical, k, metric)
    d2 = knnx_distance(empirical, postpred, k, metric)

    n, p = empirical.shape
    m    = postpred.shape[0]

    kld = np.log(m/n) + p * (np.log(d2).mean(axis = 0) - np.log(d1).mean(axis = 0))
    return kld

if __name__ == '__main__':
    pass

# EOF
