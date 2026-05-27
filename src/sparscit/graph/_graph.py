import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Self, Optional, NamedTuple

import igraph as ig
from scipy.sparse import csr_matrix
from anndata import AnnData

from .._utils import ArgAssert
from .. import _utils
from .._logging import logging


class Graph:
    """
    Lightweight graph representation using edge lists.

    Stores edges as an array of (source, target) pairs with optional weights.
    Can be converted to and from igraph objects and sparse distance matrices.

    Attributes
    ----------
    edges : np.ndarray
        Array of shape (n_edges, 2) with (source, target) pairs
    weights : np.ndarray
        Edge weights; empty array if unweighted
    obsp_key : str
        Key in ``adata.obsp`` where the original distance matrix is stored
    """
    edges: np.ndarray
    _directed: bool
    weights: np.ndarray
    _vcount: int

    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Graph with {self.vcount()} nodes, {self.ecount()} edges. Directed={str(self.is_directed())}, weighted={str(self.is_weighted())}, has_layout={str(self.has_layout())}"

    def vcount(self) -> int:
        return self._vcount

    def ecount(self) -> int:
        return self.edges.shape[0]

    def is_directed(self) -> bool:
        return self._directed

    def is_weighted(self) -> bool:
        return self.weights.shape[0] != 0

    def __init__(self, vcount: int, obsp_key: str, directed: bool = False) -> None:
        self._directed = directed
        self._vcount = vcount
        self.edges = np.array([[]])
        self.weights = np.array([])
        self.mean_nn = 0,
        self.obsp_key = obsp_key
        #self.max_degree = max_degree

    def has_layout(self) -> bool:
        return False

    def to_igraph(self) -> Any:
        """
        Returns igraph with same edges
        """

        g = ig.Graph(directed=self.is_directed())
        g.add_vertices(int(self.vcount())) 
        g.add_edges([(a[0],a[1]) for a in (self.edges)])
        if self.is_weighted():
            g.es["weight"] = (self.weights)
        else:
            g.es["weight"] = 1

        return g

    def cast_edges(self, directed: bool, self_edge: bool = True, copy: bool = False) -> Optional[np.ndarray]:
        """
        Enforce constraints on edges, discarding those that do not meet conditions.

        If the graph is directed and ``directed=False``, converts to undirected by
        merging reciprocal edges. Optionally removes self-edges.

        Parameters
        ----------
        directed
            Whether the resulting edges should be directed
        self_edge
            Whether to keep self-edges (loops); if False, self-edges are removed
        copy
            If True, return edges as an array instead of replacing in place

        Returns
        -------
        np.ndarray or None
            Edge array only if ``copy`` is True; otherwise modifies in place and returns None
        """
        edges = np.array(self.edges)
        if self.is_directed() and (directed == False):
            _g = ig.Graph(directed=True)
            _g.add_vertices(int(self.vcount())) 
            _g.add_edges([(a[0],a[1]) for a in np.array(self.edges)])
            # source,target = list(zip(*_g.as_undirected().to_tuple_list())) 
            edges = np.array(_g.as_undirected().to_tuple_list())# please rewrite

            logging.info(f"lost {self.ecount() - edges.shape[0]} edges after conversion to undirected graph")
            self._directed = False
        if self_edge == False:
            _sedge = (edges[:,0] == edges[:,1])
            if (_sedge.any()):
                logging.info(f"lost {_sedge.sum()} self-edges")
            edges = edges[~_sedge]
        if copy:
            return edges.copy()
        self.edges = edges.copy()
        return None

    def _check_graph(self, adata: 'AnnData') -> None:
        if adata.shape[0] != self.vcount():
            raise ValueError("The graph and anndata object are no longer compatible (mismatch in cell number). Please generate a new graph")

        if self.vcount() <= self.edges.max():
            raise ValueError("Graph contains edges connecting non-existent vertices")

    @staticmethod
    def _umap_compatible(
        matrix: csr_matrix
    ) -> tuple[np.ndarray, np.ndarray]:
        coo = matrix.tocoo()

        ratio = (coo.nnz / coo.shape[0])
        ArgAssert((ratio % 1) == 0, "All cells need the same number of neighbors, please don't use enn")
        ratio = int(ratio)

        ids = np.reshape(coo.col, (coo.shape[0], ratio))
        dists = np.reshape(coo.data, (coo.shape[0], ratio))
        return ids, dists

    @staticmethod
    def from_distances(
            matrix: csr_matrix,
            obsp_key: str,
            directed: bool,
            weighted: bool
    ) -> 'Graph':
        """
        Create a Graph from a sparse distance matrix.

        Parameters
        ----------
        matrix
            Sparse distance matrix (CSR format) with non-zero entries as edges
        obsp_key
            Key in ``adata.obsp`` where the distance matrix is stored
        directed
            Whether the graph should be directed. It is up to the caller to ensure
            the edges are consistent with this flag
        weighted
            Whether to store the distance values as edge weights

        Returns
        -------
        :class:`Graph`
            New graph instance with edges derived from the distance matrix
        """
        sources, targets = matrix.nonzero()
        coo = matrix.tocoo()
        ratio = (coo.nnz // coo.shape[0])

        g = Graph(matrix.shape[0], obsp_key=obsp_key, directed=directed) #, max_degree=ratio
        g.edges = np.array(list(zip(sources, targets)), dtype=np.int32)
        g.mean_nn = ratio

        if weighted:
            g.weights = np.array(matrix.data.copy(), dtype=np.float32)
        #else:
        #    g.weights = jnp.full((distances.data.shape[0],), 1, dtype=np.float32)
        return g
