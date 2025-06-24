cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import prange, parallel
from cython import nogil, inline, cfunc
import numpy as np

@cython.cfunc
@cython.returns(double)
@cython.nogil
def norm_diff(double[:] vec1, double[:] vec2):
    cdef int k
    cdef int n = vec1.shape[0]
    cdef double temp = 0.
    for k in range(n):
        temp += (vec1[k] - vec2[k]) * (vec1[k] - vec2[k])
    return sqrt(temp)

# cpdef double energy_score(
#         np.ndarray[dtype = np.float_t, ndim = 3] predictions, # nSamp, nDat, nCol
#         np.ndarray[dtype = np.float_t, ndim = 2] target,
#         ):
cpdef double energy_score(
          double[:,:,:] predictions,
          double[:,:] target,
          ):
    """
    Compute average energy score (multivariate CRPS).
    Assuming that the indexing of the target matrix is the same
      as the second axis of the predictions matrix.

    ES = \text{E}_F\lVert Y - y\rVert - \frac{1}{2}\text{E}_F\lVert Y - Y^{\prime}\rVert
    """
    cdef int i, j, k, n
    cdef int nSamp = predictions.shape[0]
    cdef int nDat  = predictions.shape[1]
    cdef int nCol  = predictions.shape[2]

    # Accuracy/GoF and Precision, Respectively
    cdef double[:] crps1 = np.zeros(nDat)
    cdef double[:] crps2 = np.zeros(nDat)
    cdef double _crps1, _crps2, temp

    with nogil, parallel(num_threads = 8):
        for n in prange(nDat):
            # Goodness of Fit
            for i in range(nSamp):
                crps1[n] = crps1[n] + norm_diff(predictions[i,n], target[n]) / nSamp
            # crps1[n] = temp / nSamp
            # Precision
            # temp = 0.
            for i in range(nSamp):
                for j in range(nSamp):
                    crps2[n] = crps2[n] + norm_diff(predictions[i,n], predictions[i,n]) / (nSamp * nSamp)
                    # temp += norm(predictions[i,n] - predictions[j,n])
            # crps2[n] = temp / (nSamp * nSamp)

    _crps1 = 0.
    _crps2 = 0.
    for n in range(nDat):
      _crps1 += crps1[n]
      _crps2 += crps2[n]

    return _crps1 + 0.5 * _crps2

# EOF
