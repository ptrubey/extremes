""" 
posterior.py

Algorithms for posterior analysis of results
"""

import numpy as np
import trees

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
    smat = np.zeros((dmat.shape[1], dmat.shape[1]))
    for s in range(dmat.shape[0]):
        smat[:] += (dmat[s] == dmat[s].T)
    return (smat / dmat.shape[0])

def minimum_spanning_trees(smat):
    graph = trees.Graph()
    edges = []
    for i in range(smat.shape[0]):
        for j in range(1,smat.shape[0]):
            edges.append((i,j,smat[i,j]))
    for edge in edges:
        graph.addEdge(*edge)
    graph.KruskalMST()
    
    pass


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
