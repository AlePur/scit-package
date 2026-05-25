import numpy as np
from scipy.cluster.hierarchy import dendrogram
from .._utils import symmetric_i, ArgAssert
from ._graph import Graph
import igraph as ig
import scipy.sparse as sps
from typing import Self
from numpy.typing import NDArray


def _vertexcluster(graph: Graph, membership: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    (sizes, edges, weights)

    """
    _graph = graph.to_igraph().as_undirected()
    _graph.es["weight"] = 1
    _graph.vs["size"] = 1
    _vcg = ig.VertexClustering(_graph,membership).cluster_graph("sum", "sum")

    _sizes = np.array(_vcg.vs["size"])
    _edges = np.array(_vcg.to_tuple_list())
    _weights = np.array(_vcg.es["weight"])
    return (_sizes, _edges, _weights)

def symmetric_skip(s: int, skip: list[int]) -> np.ndarray:
    _is = symmetric_i(s)
    return _is[~(np.isin(_is[:, 0], skip) + np.isin(_is[:, 1], skip))]  # type:ignore


class Link:
    """
    Helper class for link tree
    """
    index: int
    children: list[Self]
    size: int
    x: int
    visible: bool

    def __init__(self, i: int, s: int) -> None:
        self.index = i
        self.children = []
        self.size = s
        self.x = 0
        self.visible = False

    def ravel(self) -> list[Self]:
        l: list[Self] = []

        if (self.visible):
            return [self]

        l.extend(self.children[0].ravel())
        l.extend(self.children[1].ravel())
        return l

    def get_mask(self, ht: 'HierarchyTree') -> np.ndarray:
        return ht.masks[self.index]

    @staticmethod
    def distance(c1_2d: np.ndarray, c2_2d: np.ndarray) -> np.ndarray:
        return np.array(
            [np.abs(c2_2d[i] - c1_2d[i]) for i in range(c1_2d.shape[0])]
        )


class HierarchyTree:
    """
    Implements all cluster calculations
    """
    linkage: np.ndarray
    graph: Graph

    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Hierarchy on graph {str(self.graph)}, linkage={self.linkage.shape[0] != 0}"

    def __init__(self, graph: Graph, membership: np.ndarray) -> None:

        self.graph = graph
        self.uq_membership = np.unique(membership).shape[0]
        ArgAssert(
            self.uq_membership > 2,
            "There have to be more than two communities to run the hierarchical algorithm. Please run leiden again with higher resolution parameter"
        )

        self.bufsize = 2 * self.uq_membership - 1

        self.masks = np.zeros((self.bufsize, self.uq_membership), dtype=np.bool_)
        for i in range(self.uq_membership):
            self.masks[i][i] = True

        self.sizes, _edges, _weights = _vertexcluster(graph, membership)
        self.edges = np.zeros((self.uq_membership, self.uq_membership), dtype=np.int32)
        self.edges[_edges[:, 0], _edges[:, 1]] = _weights
        # self.edges = sps.coo_matrix(
        #    (_weights, (_edges[:,0], _edges[:,1])), shape=(self.uq_membership,self.uq_membership)
        # ).tocsr()

        self.linkage = np.array([])

    # 2.1
    def _rho(self, m1: np.ndarray, m2: np.ndarray) -> float:
        _esum = self.edges[m1][:, m2].sum() + self.edges[m2][:, m1].sum()

        return _esum / self.sizes[m1 + m2].sum()

    def rhos(self, m1_2d: np.ndarray, m2_2d: np.ndarray) -> np.ndarray:
        return np.array(
            [self._rho(m1_2d[i], m2_2d[i]) for i in range(m1_2d.shape[0])]
        )

    def show(self) -> None:
        assert self.linkage.shape[0] != 0
        dendrogram(self.linkage)

    def _get_linkage_rec(self) -> Link:
        assert (self.linkage.shape[0] != 0)

        ll = []
        for i in range(self.uq_membership):
            ll.append(Link(i, 1))

        for i in range(self.linkage.shape[0]):
            _l = self.linkage[i]
            nl = Link(i + self.uq_membership, int(_l[3]))
            nl.children = [ll[int(_l[0])], ll[int(_l[1])]]
            ll.append(
                nl
            )

        return ll[len(ll) - 1]

    def get_linkage(self) -> np.ndarray:
        if self.linkage.shape[0] == 0:
            self.calculate_linkage()

        return self.linkage

    def calculate_linkage(self) -> None:

        # dynamic
        _pointer = self.uq_membership
        _skip: list[int] = []
        _distances: list[float] = []
        _indices = symmetric_skip(_pointer, _skip)

        rhos = self.rhos(self.masks[_indices[:, 0]], self.masks[_indices[:, 1]])
        # rhos = sps.coo_matrix((rhos, (_indices[:,0],_indices[:,1])), shape=(_pointer, _pointer))
        # assert (rhos.col > rhos.row).all()

        # updated: _pointer, _skip, _indices, rhos
        while _pointer < self.bufsize:
            max_p = np.argmax(rhos)
            _distances.append(rhos[max_p])
            _old_c = _indices[max_p]
            _skip.extend(_old_c)

            # update masks
            self.masks[_pointer] = self.masks[_old_c[0]] + self.masks[_old_c[1]]

            _pointer += 1
            _indices = symmetric_skip(_pointer, _skip)

            rhos = self.rhos(self.masks[_indices[:, 0]], self.masks[_indices[:, 1]])

        msum = self.masks.sum(axis=1)
        self.linkage = np.array(
            [[_skip[i * 2], _skip[i * 2 + 1], 1 / _distances[i], msum[self.uq_membership + i]] for i in
             range(self.bufsize - self.uq_membership)],
            dtype=np.float64
        )

    def calculate_order(self) -> np.ndarray:
        root = self._get_linkage_rec()
        queue = [root]

        while len(queue) > 0:
            c = queue[0].children

            if len(c) == 2:

                queue[0].visible = False
                c[0].visible = True
                c[1].visible = True

                rr = root.ravel()
                masks = np.array([t.get_mask(self) for t in rr])

                indices = symmetric_skip(len(masks), [])
                rhos = self.rhos(masks[indices[:, 0]], masks[indices[:, 1]])

                xs1 = (queue[0].x, queue[0].x + c[0].size)
                xs2 = (queue[0].x + c[1].size, queue[0].x)

                def _set(xs: tuple[int, int]) -> None:
                    c[0].x = xs[0]
                    c[1].x = xs[1]

                def score(xs: tuple[int, int]) -> np.ndarray:
                    _set(xs)
                    ##assert np.array([t.size for t in rr]).sum() == self.uq_membership
                    coord = np.array([t.x + (t.size / 2) for t in rr])
                    return (rhos / Link.distance(coord[indices[:, 0]], coord[indices[:, 1]])).sum()

                s1 = score(xs1)
                s2 = score(xs2)
                if s1 > s2:
                    _set(xs1)

                if c[0].size < c[1].size:
                    queue.extend(c)
                else:
                    queue.extend([c[1], c[0]])

            queue.pop(0)

        rr = root.ravel()
        xs = np.array([t.x for t in rr])
        ids = np.array([t.index for t in rr])
        return ids[np.argsort(xs)]
