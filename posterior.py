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

def emergent_clusters(graph, k):
    """
    Emergent Clusters
    ---
    Given similarity matrix between observations, creates labels for (k-1) 
    emergent clusters.
    ---
    inputs:
        smat (n x n array)
        k    (integer)
    outputs:
        labels (n)
    """
    graph_ = graph.iloc[:-(k-1)] # delete k-1 largest edges; tree in k clusters
    N = graph[['node1','node2']].max().max() + 1 # total # of nodes 

    sets = [set() for _ in range(k)] # output sets

    todo = graph_[['node1','node2']].values.tolist() # list of edges
    done = []
    for i in range(k):
        current = todo.pop() # start the set with the first available edge.
        sets[i].add(current[0])
        sets[i].add(current[1])
        addlfound = False  # if any additional nodes were found during this sweep
        while len(todo) > 0:
            # keep running until every node in set has been found.
            while len(todo) > 0:
                current = todo.pop() # take available edge
                if current[0] in sets[i]:  # if either node in set, add other node
                    sets[i].add(current[1])
                    addlfound = True
                elif current[1] in sets[i]:
                    sets[i].add(current[0])
                    addlfound = True
                else:                # otherwise, append edge to outstack
                    done.append(current)
            if addlfound:  # if any additional nodes were found this sweep
                todo = done  # cycle out-stack to in-stack
                done = []   
                addlfound = False
            raise
        
        todo = done # cycle out-stack to in-stack, proceed to next set.
        done = []
       
    labels = np.zeros(N, dtype = int)
    for i, s in enumerate(sets):
        labels[s] = i
    return labels

if __name__ == '__main__':
    import pandas as pd
    # knock off 5 edges    
    graph = pd.read_csv('./datasets/slosh/sloshltd_mst.csv')
    labels = emergent_clusters(graph, 7)
    raise
    
    
# EOF
