cimport numpy as np
import numpy as np
# from scipy.stats import multinomial
from numpy.random import multinomial

cpdef np.ndarray[dtype = np.int_t, ndim = 1] counter(
        np.ndarray[dtype = np.int_t, ndim = 1] delta,
        int stop_point
        ):
    """
    Counts integers that appear in an array up to a stop point.  E.g.,
    an array delta = (1,1,2,3,4,0), with stop point 5 will return (1,2,1,1,1)

    Giving a stop point <= delta.max() is bad juju.
    """
    cdef np.ndarray[dtype = np.int_t, ndim = 1] counts = np.zeros(stop_point, dtype = np.int)
    cdef int iter, ndat
    ndat = delta.shape[0]
    for iter in range(ndat):
        counts[delta[iter]] += 1
    return counts

cpdef int first_zero(np.ndarray[dtype = np.int_t, ndim = 1] arr):
    """ Finds the index of the first zero in an integer array """
    cdef int arr_end = arr.shape[0]
    cdef int i
    for i in range(arr_end):
        if not arr[i]:
            return i
    else:
        return arr_end
    pass

cpdef np.ndarray[dtype = np.int_t, ndim = 1] generate_indices(
        np.ndarray[dtype = np.float_t, ndim = 1] probs,
        int n
        ):
    cdef np.ndarray[dtype = np.int_t, ndim = 1] indices = np.empty(n, dtype = np.int)
    cdef int iter_result, iter_indices
    cdef np.ndarray[dtype = np.int_t, ndim = 1] result = multinomial(n = n, pvals = probs)
    iter_indices = 0
    for iter_result in range(result.shape[0]):
        while result[iter_result] > 0:
            indices[iter_indices] = iter_result
            result[iter_result] -= 1
            iter_indices += 1
    return indices

cpdef np.ndarray[dtype = np.int_t, ndim = 1] cluster_size_matrix_slow(
        np.ndarray[dtype = np.int_t, ndim = 1] row,
        ):
    cdef:
        np.ndarray[dtype = np.int_t, ndim = 2] dmat = \
            np.zeros((row.shape[0], row.max() + 1), dtype = np.int)
        int i, j
    for i in range(row.shape[0]):
        dmat[i, row[i]] = 1
    return dmat.T @ dmat.sum(axis = 1)

cpdef np.ndarray[dtype = np.int_t, ndim = 1] cluster_size_matrix(
        np.ndarray[dtype = np.int_t, ndim = 1] row,
        ):
    cdef:
        np.ndarray[dtype = np.int_t, ndim = 1] counts
        np.ndarray[dtype = np.int_t, ndim = 1] out
        int i
    counts = counter(row, row.max() + 1)
    out    = np.empty(row.shape[0], dtype = np.int)
    for i in range(row.shape[0]):
        out[i] = counts[row[i]]
    return out

cpdef np.ndarray[dtype = np.float_t, ndim = 2] cluster_size_summary(
        np.ndarray[dtype = np.int_t, ndim = 2] delta,
        ):
    cdef:
        np.ndarray[dtype = np.int_t, ndim = 2] working = \
                np.empty((delta.shape[0], delta.shape[1]), dtype = np.int)
        np.ndarray[dtype = np.float_t, ndim = 2] out = np.empty((3, delta.shape[1]))
        int i
    for i in range(delta.shape[0]):
        working[i] = cluster_size_matrix(delta[i])
    out[0] = working.mean(axis = 0)
    out[1] = working.std(axis = 0)
    out[2] = np.exp(np.log(working).mean(axis = 0))
    return out

cpdef np.ndarray[dtype = np.float_t, ndim = 2] row_neighbor_matrix(
        np.ndarray[dtype = np.int_t, ndim = 1] row,
        ):
    """ For a given row of delta, return a matrix of delta coincidence """
    cdef:
        np.ndarray[dtype = np.int_t, ndim = 2] dmat = \
              np.zeros((row.shape[0], row.max() + 1), dtype = np.int)
        int i, j
    for i in range(row.shape[0]):
        dmat[i,row[i]] = 1
    return dmat @ dmat.T

cpdef np.ndarray[dtype = np.float_t, ndim = 2] find_neighbors(
        np.ndarray[dtype = np.int_t, ndim = 2] delta,
        ):
    cdef:
        np.ndarray[dtype = np.float_t, ndim = 2] weight = np.zeros((delta.shape[1], delta.shape[1]))
        int i
    for i in range(delta.shape[0]):
        weight += row_neighbor_matrix(delta[i])
    return weight / delta.shape[0]

# EOF
