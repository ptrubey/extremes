cimport numpy as np
import numpy as np
from numpy.random import uniform
from libc.math cimport log, exp

cpdef double univariate_slice_sample(object logd, double starter, double increment_size = 1.):
    cdef double bound_logd, lb, ub, try_value

    bound_logd = log(uniform(0, exp(logd(starter))))
    lb = starter - increment_size
    ub = starter + increment_size

    while logd(lb) > bound_logd:
        lb -= increment_size
    while logd(ub) > bound_logd:
        ub += increment_size

    try_value = lb + uniform() * (ub - lb)

    while logd(try_value) < bound_logd:
        if try_value > starter:
            ub = try_value
        else:
            lb = try_value

        try_value = lb + uniform() * (ub - lb)

    return try_value

cpdef double skip_univariate_slice_sample(object logd, double starter, double increment_size = 1., int skips = 2):
    cdef double try_value
    cdef int _
    try_value = starter
    for _ in range(skips):
        try_value = univariate_slice_sample(logd, try_value, increment_size)
    return try_value

# EOF
