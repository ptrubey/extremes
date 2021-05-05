import numpy as np
from multiprocessing import Pool
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

def energy_score(predictions, targets):
    # pool = Pool(processes = 12)
    # res1 = pool.map(prediction_pairwise_distance, predictions)
    # res2 = pool.map(target_pairwise_distance, zip(predictions, targets))
    # pool.close()
    res1 = map(prediction_pairwise_distance, predictions)
    res2 = map(target_pairwise_distance, zip(predictions, targets))
    return np.array(list(res2)).mean() - 0.5 * np.array(list(res1)).mean()

def intrinsic_energy_score(dataset):
    res1 = prediction_pairwise_distance(dataset) # same for all elements of df.  only do once.
    pool = Pool(processes = 12)
    res2 = pool.map(target_pairwise_distance, zip(repeat(dataset),dataset))
    return np.array(list(res2)).mean() - 0.5 * res1

if __name__ == '__main__':
    pass

# EOF
