""" slice sampling functions """
from numpy.random import uniform
from math import log, exp

def univariate_slice_sample(logd, starter, increment_size):
    bound_logd = log(uniform(0, exp(logd(starter))))

    bounds = [starter - increment_size, starter + increment_size]
    while logd(bounds[0]) > bound_logd:
        bounds[0] -= increment_size
    while logd(bounds[1]) > bound_logd:
        bounds[1] += increment_size

    try_value = bounds[0] + uniform() * (bounds[1] - bounds[0])
    while logd(try_value) < bound_logd:
        if try_value > starter:
            bounds[1] = try_value
        else:
            bounds[0] = try_value
        try_value = bounds[0] + uniform() * (bounds[1] - bounds[0])
    return try_value

def skip_univariate_slice_sample(logd, starter, increment_size, skips):
    try_value = starter
    for _ in range(skips):
        try_value = univariate_slice_sample(logd, try_value, increment_size)
    return try_value

# EOF
