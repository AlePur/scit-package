# distutils: language = c++

cimport numpy as np
from libc.math cimport sqrt, log
import scipy as osc
import numpy as onp

ctypedef unsigned char uint8_t
ctypedef float float32_t
ctypedef double float64_t
ctypedef char int8_t
ctypedef unsigned int uint32_t
ctypedef unsigned long long uint64_t
ctypedef Py_ssize_t intp_t

def mannwhitneyu_from_summary(
        uint32_t n1,
        uint32_t n2,
        float64_t R1,
        float64_t R2
):
    cdef float32_t u1, u2
    cdef float32_t u_statistic, mean_u, std_u, z,

    # Calculate U statistics
    u1 = R1 - <float32_t>(n1 * (n1 + 1)) / 2
    u2 = R2 - <float32_t>(n2 * (n2 + 1)) / 2

    # Use smaller U value by convention
    u_statistic = min(u1, u2)

    # For normal approximation
    mean_u = <float32_t>(n1 * n2) / 2
    std_u = sqrt(<float32_t>(n1 * n2 * (n1 + n2 + 1)) / 12)

    # Calculate z-score
    z = <float32_t>(u_statistic - mean_u) / std_u

    # Calculate two-sided p-value

    return log(u1 / mean_u), 2 * osc.stats.norm.cdf(-abs(z))


def ranks_from_sparse(
        sparse_data_group1,
        sparse_data_group2
):
    cdef intp_t n1, n2
    cdef np.ndarray[float, ndim=1] combined
    cdef np.ndarray[double, ndim=1] ranks
    cdef np.ndarray[float, ndim=1] data1 = sparse_data_group1.data
    cdef np.ndarray[float, ndim=1] data2 = sparse_data_group2.data

    # Get sample sizes
    n1 = data1.shape[0]
    n2 = data2.shape[0]

    # Combine data for ranking
    combined = onp.concatenate([data1, data2])

    # Compute ranks of the combined data
    ranks = osc.stats.rankdata(combined)
    return n1, n2, ranks[:n1].sum(), ranks[n1:].sum()