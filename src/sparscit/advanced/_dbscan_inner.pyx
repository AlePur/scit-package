# distutils: language = c++
#
# Fast inner loop for DBSCAN.

from libcpp.vector cimport vector

ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef unsigned long long uint64_t
ctypedef Py_ssize_t intp_t
# Author: Lars Buitinck
# License: 3-clause BSD
def dbscan_inner(const uint8_t[::1] is_core,
                 list neighborhoods,
                 intp_t[::1] labels):
    cdef intp_t i, label_num = 0, v
    cdef uint32_t[:] neighb
    cdef vector[intp_t] stack

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # This is very similar to the classic algorithm for computing connected
        # components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        while True:
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:
                    neighb = neighborhoods[i]
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        if labels[v] == -1:
                            stack.push_back(v)

            if stack.size() == 0:
                break
            i = stack.back()
            stack.pop_back()

        label_num += 1