from anndata import AnnData
from .._logging import logging
import numpy as np
from ._neighbors import _neighbor_graph
import pandas as pd
from .._utils import ArgAssert, _get_membership
from ..graph._hierarchy import HierarchyTree
import random

from .._logging import warn_overwrite

import igraph as ig

def community_hierarchy(
        adata: AnnData,
        community_key: str = 'leiden',
        *,
        obsp_key: str | None = None
) -> None:
    """\
    Construct a hierarchical community tree based on previous flat-clustering.
    Implementation from Dreveton et al `arXiv:2306.00833v2 [cs.SI]`.
    This also calculates the optimal ordering of the clusters.

    Parameters
    ----------
    adata
        AnnData object with neighbor graph in `.obsp`
    community_key
        Key in `.obs` for community memberships
    obsp_key
        Key in `.obsp` for connectivity matrix. If None, uses 'connectivities'

    Returns
    -------
    `adata.uns['community_tree']['base']`
        Number of original clusters
    `adata.uns['community_tree']['linkage']`
        Linkage matrix that can be plotted with :func:`sparscit.adv.pl.community_dendrogram` or with :func:`scipy.cluster.hierarchy.dendrogram`
    `adata.uns['community_tree']['order']`
        Optimal order of clusters (minimized linear edge distance)
    `adata.uns['community_tree']['obs_key']`
        Key in `.obs` identifying cluster membership on single cell resolution

    """
    graph = _neighbor_graph(adata, obsp_key=obsp_key)
    graph._check_graph(adata)

    hier = HierarchyTree(graph, _get_membership(adata, community_key))

    adata.uns['community_tree'] = {}
    adata.uns['community_tree']['base'] = hier.uq_membership
    adata.uns['community_tree']['linkage'] = hier.get_linkage()
    # order is the x coord from 0 - max with the values being ID, not the other way around
    adata.uns['community_tree']['order'] = hier.calculate_order()
    adata.uns['community_tree']['obs_key'] = community_key

def leiden(
        adata: AnnData,
        resolution: float = 1.0,
        *,
        obsp_key: str = 'connectivities',
        random_seed: int | None = None,
        return_raw: bool = False,
        beta: float = 0.01
) -> None | np.ndarray:
    """\
    Leiden community detection algorithm using `igraph` implementation

    Parameters
    ----------
    adata
        AnnData object with neighbor graph in `.obsp`
    resolution
        Resolution of leiden algorithm. Higher resolutions lead to more smaller communities,
        while lower resolutions lead to fewer larger communities.
    obsp_key
        Key in `.obsp` for connectivity matrix
    random_seed
        Random seed for leiden
    return_raw
        Return leiden membership as array instead of saving to AnnData
    beta
        Beta parameter for leiden. More info in igraph community_leiden documentation

    Returns
    -------
    `adata.obs['leiden']`
        Community membership
    """
    graph = _neighbor_graph(adata, obsp_key=obsp_key)
    graph._check_graph(adata)
    _d = adata.obsp[graph.obsp_key].copy()

    _g: 'ig.Graph' = graph.to_igraph().as_undirected()
    source, target = list(zip(*_g.to_tuple_list()))
    _weights = np.c_[np.array(_d[source, target])[0], np.array(_d[target, source])[0]]
    _weights = np.max(_weights, axis=1)

    random.seed(random_seed)
    ig.set_random_number_generator(random)
    vc = _g.community_leiden(
        objective_function='modularity',
        weights=np.array(_weights).astype(np.float64),
        resolution=resolution,
        beta=beta,
        n_iterations=2
    )

    if return_raw:
        return np.array(vc.membership)
    adata.obs['leiden'] = pd.Categorical(np.array(vc.membership))
