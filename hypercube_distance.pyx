import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cdef int argmax(double[:] x):
    cdef int i, m
    cdef double buf = x[0]
    for i in range(x.shape[0]):
        if x[i] > buf:
            m = i
            buf = x[i]
    return m

cdef double vector_norm(double[:] x, double[:] y):
    cdef:
        int i
        double s = 0.
    for i in range(x.shape[0]):
        s += (x[i] - y[i]) * (x[i] - y[i])
    return sqrt(s)

cdef double hdist(double[:] x, double[:] y):
    cdef:
        int starting_face, ending_face
    starting_face = argmax(x)
    ending_face = argmax(y)
    if starting_face == ending_face: # If they're on the same face, return euclidean norm
        return vector_norm(x, y)
    return 0.

cpdef double[:] pairwise_distance_internal(double[:,:] x):
    pass

cpdef double[:] pairwise_distance_comparison(double[:,:] x, double[:] y):
    pass


# EOF
