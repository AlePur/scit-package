from numbers import Integral, Real

import numpy as np
import scipy.sparse as sps
import warnings

from ._dbscan_inner import dbscan_inner

class SubgraphDBSCAN:
    """Perform DBSCAN clustering from vector array or distance matrix."""

    def __init__(
            self,
            enrich: float = 0.1,
            min_degree: int = 0,
            min_samples: int = 0,
            min_active: int | None = None
    ):
        self.min_degree = min_degree
        self.min_samples = min_samples
        self.enrich = enrich
        self.min_active = min_active

    def fit(self, X: sps.csr_matrix, degrees: np.ndarray | None, mask: np.ndarray | list):
        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)

        # set the diagonal to explicit values, as a point is its own
        # neighbor
        mask = np.array(mask, dtype=np.bool_)

        A = X[mask][:, mask]

        if degrees is not None:
            new_degrees = A.sum(axis=1).A1
            enrichment = new_degrees / degrees[mask]
        neighborhoods = []
        for i in range(A.shape[0]):
            neighborhoods.append(
                A.indices[A.indptr[i]:A.indptr[i+1]].astype(np.uint32)
            )
        n_neighbors = np.array([n.shape[0] for n in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full((mask.sum(),), -1, dtype=np.intp)

        # A list of all core samples found.
        if degrees is not None:
            core_samples = np.asarray(
                ((degrees[mask] >= self.min_degree) *
                 (n_neighbors >= self.min_samples) *
                 (enrichment >= self.enrich)).astype(np.bool_)
                , dtype=np.uint8
            )
        else:
            core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)
        if core_samples.sum() == 0:
            self.labels_ = labels
            return self

        dbscan_inner(core_samples, neighborhoods, labels)

        # Remove non-core samples
        labels[~core_samples.astype(np.bool_)] = -1

        if self.min_active is not None:
            ci, cn = np.unique(labels, return_counts=True)
            active_filtered = ci[(cn > self.min_active) * (ci != -1)]

            filtered_labels = np.full(labels.shape, -1, dtype=np.int32)
            for i, af in enumerate(active_filtered):
                filtered_labels[labels == af] = i
            labels = filtered_labels

        #self.core_samples_ = core_samples.astype(np.bool_)
        self.labels_ = labels
        return self
