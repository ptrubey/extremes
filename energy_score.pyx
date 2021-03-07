cimport numpy as np
cimport math
import numpy as np

cpdef float es(
        np.ndarray[dtype = np.float_t, ndim = 3] predictions, # nSamp, nDat, nCol
        np.ndarray[dtype = np.float_t, ndim = 2] target,
        ):
    """
    Compute average energy score (multivariate CRPS).
    Assuming that the indexing of the target matrix is the same
      as the second axis of the predictions matrix.

    ES = \text{E}_F\lVert Y - y\rVert - \frac{1}{2}\text{E}_F\lVert Y - Y^{\prime}\rVert
    """
    cdef int i, j, k, n
    cdef nSamp = predictions.shape[0]
    cdef nDat  = predictions.shape[1]
    cdef nCol  = predictions.shape[2]
    cdef float temp_norm = 0.
    cdef float temp_hold = 0.

    # Accuracy/GoF and Precision, Respectively
    cdef np.ndarray[dtype = np.float64, ndim = 1] crps1
    cdef np.ndarray[dtype = np.float64, ndim = 2] crps2

    crps1 = np.zeros(nDat)
    crps2 = np.zeros(nDat)

    for n in range(nDat):
        # Goodness of Fit

        for i in range(nSamp):
          temp = 0.
          for k in range(nCol):
            temp += (predictions[i,n,k] - target[n,k]) * (predictions[i,n,k] - target[n,k])
        crps1[n] = math.sqrt(temp) / nSamp

        # Precision
        temp = 0.
        for i in range(nSamp):
          for j in range(nSamp):
            for k in range(nSamp):







    return

# EOF
