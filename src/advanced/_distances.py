import numpy as np
import polars as pl
from anndata import AnnData
from typing import Any, Callable, Literal
import os
import scipy.sparse as sps
from sklearn.cluster import MiniBatchKMeans

from .._utils import _csr_mean, ArgAssert, _double_lambda, _get_memberships
from .._logging import logging
from ..tools._layers import LayerConfig, _format_var_names
from ._dbscan import SubgraphDBSCAN
from ._denoise import denoise_inner

def classify_denoised_defunct(
        adata: AnnData,
        layer_config: LayerConfig,
        embedding_key: str,
        min_degree: int,
        min_active: int,
        *,
        mode: Literal['top_layer', 'threshold'] = 'threshold',
        rep_min_cells: int = 100,
        exclude_below_rep: int = 0,
        obsp_key: str = 'distances'
) -> None:
    """
    Classify active cell clusters.

    Parameters
    ----------
    adata
    LayerConfig
        Result of denoised data
    min_degree
        Exclude points with degree less than min_degree
    min_active
        Exclude active clusters with less than min_active members
    mode
        Whether pick active cells based on relative replication level, or absolute replication threshold
    obsp_key
        Key for distances in obsp
    subcluster_per_n_cells
        Number of cell in each Kmeans subcluster
    rep_min_cells
        Minimum cells for accepting replication level in activity layer
    exclude_below_rep
        Exclude features below this replication level

    Returns
    -------

    """
    data = layer_config.transform(adata, raw_view=True)
    datamax = int(data.max())
    if mode == 'top_layer':
        # cells in each level
        levels = np.array([((data == (1 + l)).sum(axis=0)).A1 for l in range(datamax)]).T
        levels = np.cumsum(levels[:, ::-1], axis=1)
        # first occurrence
        picked_level = datamax - (levels >= rep_min_cells).argmax(axis=1)
        # check max sum if above th
        layer_config.cached_mask = levels[:, -1] >= rep_min_cells
        if exclude_below_rep > 0:
            layer_config.cached_mask = layer_config.cached_mask * (picked_level >= exclude_below_rep)
        data = sps.csr_matrix(data[:, layer_config.cached_mask] >= picked_level[layer_config.cached_mask])
    else:
        td = data.copy()
        td.data = td.data >= exclude_below_rep
        layer_config.cached_mask = td.sum(axis=0).A1 > rep_min_cells
        data = sps.csr_matrix(td[:, layer_config.cached_mask])
    data = data.astype(np.bool_)
    find = FindActive(adata, layer_config, (adata.shape[0], 1), save_prefix='active_class_', override_data=data)

    logging.info(
        f"Running subclassification on {find.var_mask.sum()} "
        f"genes, excluding below degree of {min_degree}"
    )

    X_emb = adata.obsm[embedding_key]
    G = adata.obsp[obsp_key].copy()
    _G = G.copy().tocsr()

    def _classify_subset(
            clusters: np.ndarray, classifier_args: dict, bind: list[tuple[str, Callable]],
            mode: str, *, full: Any = -1, ignore: Any = None
    ) -> np.ndarray | None:
        sub_clusters = np.full((adata.shape[0],), full, dtype=np.int32)

        uq = np.unique(clusters)
        if ignore is not None:
            uq = uq[uq != ignore]
        if uq.shape[0] == 0:
            return None

        for _uq in uq:
            ac = clusters == _uq

            for b in bind:
                classifier_args[b[0]] = b[1](ac)

            if mode == 'DBSCAN':
                classifier = SubgraphDBSCAN(
                    **classifier_args
                ).fit(G, None, ac)
            else:
                classifier = MiniBatchKMeans(
                    **classifier_args
                ).fit(X_emb[ac])

            sub_clusters[ac] = classifier.labels_ + sub_clusters.max() + 1
        return sub_clusters

    def run_internal(
            _self: FindActive,
            gene_index: np.ndarray
    ) -> None:
        # x > t_rna
        pass_th_x = _self.Z[:, gene_index].todense().A1
        if pass_th_x.sum() != 0:

            # HDBSCAN(min_cluster_size=min_points)
            # OPTICS(min_samples=min_points, cluster_method='dbscan', max_eps=1.0)
            active = _classify_subset(
                pass_th_x,
                {
                    "min_samples": min_degree,
                    "min_active": min_active
                },
                [],
                mode="DBSCAN",
                ignore=False
            )
            #
            active_sub_clusters = _classify_subset(
                active,
                {},
                [(
                    "n_clusters",
                    lambda ac: np.clip(int(ac.sum() / subcluster_per_n_cells), 1, 200)
                )],
                mode="KMeans",
                ignore=-1,
            )

            if active_sub_clusters is not None:
                #if not (active_sub_clusters == -1).all():
                _self.data[_self.genes_to_scan[gene_index]] = sps.csr_array(active_sub_clusters + 1)
                _self.data[_self.genes_to_scan[gene_index]].eliminate_zeros()

    mla = find.run(run_internal)
    k = mla.future_key
    adata.obsm[k] = mla.to_sps().T.astype(np.uint16)
    logging.info(f'Added {k} to .obsm')


def classify_enrichment(
        adata,
        layer_config: LayerConfig,
        enrich_a: float,
        enrich_b: float,
        enrich_g: float,
        q: float
):
    # s > a + b(mean)
    # (s - a) / mean > b
    X = layer_config.transform(adata, use_cached_mask=False).todense()
    # mean = X.mean(axis=0).A1
    base = np.quantile(np.asarray(X),q,axis=0)
    
    s = f'{layer_config.layer}_active'
    logging.info(f'Added to .obsm["{s}"]')
    #R = X / enrich_a
    #X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    #adata.obsm[s] = sps.csr_matrix((R + (X > enrich_b)) > 1)
    #adata.obsm[s] = sps.csr_matrix(((X - enrich_a) @ sps.diags(1.0 / (np.sqrt(mean+1)-1))) > enrich_b)
    adata.obsm[s] = sps.csr_matrix(((X - enrich_a) @ sps.diags(1.0 / (np.power(base+1,enrich_g)-1))) > enrich_b)

def neighborhood_activities(
        adata: AnnData,
        layer_config: LayerConfig,
        *,
        community_matrix: str = 'leiden_matrix',
        use_cached_mask: bool = False
) -> sps.csr_matrix | None:
    ArgAssert(community_matrix in adata.obsm.keys(), f'{community_matrix} not found in .obsm')
    communities: np.ndarray = adata.obsm[community_matrix].T.astype(np.uint32)
    
    L, B = list_of_arrays_to_csr_with_samples(communities)
    X = layer_config.transform(adata, use_cached_mask=use_cached_mask, binarize=True).astype(np.uint32)
    # norm
    D = sps.diags(1.0 / L.astype(np.float32).sum(axis=0).A1)

    s = f'neighborhood_{layer_config.layer}'
    sn = f'neighborhood_{layer_config.layer}_names'
    logging.info(f'Added to .obsm["{s}"], .uns["{sn}"]')
    adata.uns[sn] = layer_config.get_feature_names(use_cached_mask=use_cached_mask)
    # AC = (L @ D @ (L.T @ X))
    # AC.data = (AC.data - AC.data.min()) / (AC.data.max() - AC.data.min())
    adata.obsm[s] = (L @ D @ (L.T @ X)) 


def array_to_sparse_bool_matrix(arr):
    """
    AI GENERATED:
    Convert 1D array of integers to sparse boolean matrix.
    Assumes integers are 0, 1, 2, ..., m-1 where m is max(arr) + 1.
    
    Parameters:
    -----------
    arr : array-like, shape (n,)
        Input array of integers starting from 0
    
    Returns:
    --------
    sparse_matrix : scipy.sparse.csr_matrix, shape (n, m)
        Sparse boolean matrix where m = max(arr) + 1
    """
    arr = np.asarray(arr)
    n = len(arr)
    m = arr.max() + 1
    row_indices = np.arange(n)
    col_indices = arr
    data = np.ones(n, dtype=bool)
    sparse_matrix = sps.csr_matrix((data, (row_indices, col_indices)), 
                                    shape=(n, m), dtype=bool)
    return sparse_matrix


def list_of_arrays_to_csr_with_samples(list_of_arrays):
    """
    AI GENERATED:
    Convert list of arrays to horizontally stacked CSR matrix and a sample mapping matrix.
    
    Parameters:
    -----------
    list_of_arrays : list of array-like
        List of l arrays, each of shape (n,), containing integer values
    
    Returns:
    --------
    stacked_matrix : scipy.sparse.csr_matrix, shape (n, k)
        Horizontally stacked sparse boolean matrix where k is the sum of unique 
        integers across all samples. Each row corresponds to one position in the arrays.
    
    sample_matrix : scipy.sparse.csr_matrix, shape (k, l)
        Matrix indicating which sample (column) each feature (row) came from.
        Each row has exactly one True value indicating its source sample.
    
    Example:
    --------
    >>> arr1 = np.array([0, 1, 2])  # 3 unique values (0,1,2)
    >>> arr2 = np.array([1, 2, 3])  # 3 unique values (1,2,3)
    >>> stacked, samples = list_of_arrays_to_csr_with_samples([arr1, arr2])
    >>> stacked.shape  # (3, 6) - n=3, 6 total unique values
    >>> samples.shape  # (6, 2) - 6 features, 2 samples
    """
    
    l = len(list_of_arrays)  # number of samples

    _a = []
    sizes = []
    for community in list_of_arrays:
        m = array_to_sparse_bool_matrix(community)
        _a.append(m)
        sizes.append(m.shape[1])
    L = sps.hstack(_a)

    # Create sample_matrix (k, l) using sizes
    k = sum(sizes)
    row_indices = []
    col_indices = []
    
    for sample_idx, size in enumerate(sizes):
        row_indices.extend(range(sum(sizes[:sample_idx]), sum(sizes[:sample_idx]) + size))
        col_indices.extend([sample_idx] * size)
    
    data = np.ones(len(row_indices), dtype=bool)
    sample_matrix = sps.csr_matrix((data, (row_indices, col_indices)), 
                                    shape=(k, l), dtype=bool)
    
    assert np.all(sample_matrix.sum(axis=0).A1 == np.array(sizes))
    return L, sample_matrix