""" 
posterior.py

Algorithms for posterior analysis of results
"""

import numpy as np
import trees

EPS = 1e-16

def similarity_matrix(dmat):
    """
    Similarity Matrix
    ---
    Posterior Similarity Matrix given stochastic cluster assignments in Bayesian 
    Non-parametric mixture model.
    ---
    inputs:
        dmat (s x n)
    outputs:
        smat (n x n)    
    """
    dmatI = dmat.T
    smat = np.zeros((dmat.shape[0], dmat.shape[0]))
    for s in range(dmat.shape[1]):
        smat[:] += (dmatI[s][None] == dmatI[s][:,None])
    return (smat / dmat.shape[1])

def minimum_spanning_trees(smat):
    graph = trees.Graph(smat.shape[0])
    edges = []
    for i in range(smat.shape[0]):
        for j in range(i + 1, smat.shape[0]):
            # edge weight is reciprocal of posterior co-clustering prob.
            edges.append((i,j, 1 / (smat[i,j] + EPS)))
    for edge in edges:
        graph.addEdge(*edge)
    return graph.KruskalMST()

def emergent_clusters(smat):
    """
    Emergent Clusters
    ---
    Given similarity matrix between observations, creates labels of emergent
    clusters.
    ---
    inputs:
        smat (n x n)
    outputs:
        labels (n)
    """



# EOF
