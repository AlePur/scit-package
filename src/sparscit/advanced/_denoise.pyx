# distutils: language = c++

cimport numpy as np
import numpy as onp

ctypedef unsigned char uint8_t
ctypedef char int8_t
ctypedef unsigned int uint32_t
ctypedef unsigned long long uint64_t
ctypedef Py_ssize_t intp_t

def denoise_inner(
        np.ndarray[uint8_t, ndim=2] res,
        np.ndarray[uint32_t, ndim=2] communities,
        np.ndarray[uint8_t, ndim=1] activity,
        const float enrichment_threshold
):
    cdef intp_t i, j
    cdef np.ndarray[uint8_t, ndim=1] omask
    cdef np.ndarray[uint32_t, ndim=1] uq_cats

    for i in range(res.shape[0]):
        uq_cats = onp.unique(communities[i])
        for j in range(uq_cats.shape[0]):
            omask = communities[i] == uq_cats[j]
            res[i][omask] = activity[omask].sum() > (enrichment_threshold * omask.sum())
    return