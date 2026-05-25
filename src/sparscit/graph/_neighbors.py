import numpy as np
from scipy.sparse import csr_matrix, diags, spdiags
from anndata import AnnData
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer

from .._utils import Metric, ArgAssert
from .._settings import settings
from ..graph._graph import Graph
from .._logging import logging
from ..tools._metadata import add_metadata
from umap.umap_ import fuzzy_simplicial_set


def _neighbor_graph(
        adata: AnnData,
        directed: bool = True,
        *,
        obsp_key: str | None = None
) -> Graph:
    """\
    Make Graph from previously calculated kNN in `.obsp['distances']`

    Parameters
    ----------
    adata
        AnnData object with distances in `.obsp`
    directed
        Whether graph will be directed or not
    obsp_key
        Key in `.obsp` to use for cell-to-cell connectivities

    Returns
    -------
    :class:`Graph`
        A Graph object built from the connectivity matrix

    """
    if obsp_key is None:
        obsp_key = 'connectivities'
    ArgAssert(obsp_key in adata.obsp.keys(), "no obsp_key found in .obsp")

    g = Graph.from_distances(adata.obsp[obsp_key], obsp_key, True, True)
    if not directed:
        g.cast_edges(directed=False)
    return g


def enn(
        adata: AnnData,
        embedding_key: str,
        epsilon: float = 1,
        *,
        metric: Metric = "euclidean"
) -> None:
    """\
    Calculate epsilon nearest neighbors using :class:`sklearn.neighbors.RadiusNeighborsTransformer`

    Parameters
    ----------
    adata
        AnnData object with embedding in `.obsm`
    embedding_key
        Key in `.obsm` for the embedding to use
    epsilon
        Radius of neighborhood for including neighbors
    metric
        Distance metric

    Returns
    -------
    `adata.obsp['epsilon_distances']`
        Sparse distance matrix of epsilon-nearest neighbors
    """

    ArgAssert(embedding_key in adata.obsm.keys(), "embedding_key not in .obsm")

    X = adata.obsm[embedding_key]

    transformer = RadiusNeighborsTransformer(
        mode='distance',
        radius=epsilon,
        algorithm='auto',
        metric=metric,
        n_jobs=settings.n_jobs
    )

    _distances: csr_matrix = transformer.fit_transform(X)
    _distances.setdiag(0)
    _distances.eliminate_zeros()

    adata.obsp['epsilon_distances'] = _distances


def knn(
        adata: AnnData,
        embedding_key: str,
        *,
        add_connectivities: bool = True,
        n_neighbors: int = 10,
        metric: Metric = "euclidean"
) -> None:
    """\
    Calculate k-nearest neighbors using :class:`sklearn.neighbors.KNeighborsTransformer`.
    In addition, calculate connectivities using :func:`UMAP.fuzzy_simplicial_set`

    Parameters
    ----------
    adata
        AnnData object with embedding in `.obsm`
    embedding_key
        Key in `.obsm` for the embedding to use
    add_connectivities
        Whether to calculate connectivities and save them in `.obsp`
    n_neighbors
        Number of neighbors to calculate
    metric
        Distance metric

    Returns
    -------
    `adata.obsp['distances']`
        Sparse distance matrix of k-nearest neighbors
    `adata.obsp['connectivities']`
        Fuzzy simplicial set connectivities (only if `add_connectivities=True`)
    """

    ArgAssert(embedding_key in adata.obsm.keys(), "embedding_key not in .obsm")

    X = adata.obsm[embedding_key]

    transformer = KNeighborsTransformer(
        mode='distance',
        n_neighbors=n_neighbors,
        algorithm='auto',
        metric=metric,
        n_jobs=settings.n_jobs
    )
    _distances: csr_matrix = transformer.fit_transform(X)  # CSR matrix apparently

    assert (_distances.getnnz(axis=1) == n_neighbors + 1).all()

    _mask = np.full((X.shape[0], n_neighbors + 1), True)
    _mask[:, 0] = False
    _mask = _mask.ravel()

    indices = _distances.indices[_mask]
    distances = _distances.data[_mask]
    indptr = np.arange(0, indices.shape[0] + 1, n_neighbors)

    _distances = csr_matrix(
        (
            distances,
            indices,
            indptr,
        ),
        shape=(X.shape[0], X.shape[0]),
    )

    if add_connectivities:
        ids, dists = Graph._umap_compatible(_distances)
        _simplex_set = fuzzy_simplicial_set(X, n_neighbors, 0, metric, knn_indices=ids, knn_dists=dists)[0]
        adata.obsp['connectivities'] = _simplex_set.tocsr()

    adata.obsp['distances'] = _distances


def knn_impute(
        adata: AnnData,
        layers: list[str],
        impute_strength: float = 0.6,
        *,
        density_normalize: bool = True,
        obsp_key: str,
        normalize_with_obs_counts: bool = True
) -> None:
    """
    Impute data using k-nearest neighbors.

    Parameters
    ----------
    adata
        AnnData object with neighbor graph in `.obsp`
    layers
        List of layer names to impute
    impute_strength
        Influence of neighboring cell counts on each cell. This value reflects how many
        times influence the imputed counts have compared to original counts. For example,
        impute_strength = 2.0 implies imputed values account for 2/3 of final counts while
        1/3 is the original cell counts
    density_normalize
        Whether to density-normalize the connectivity matrix before imputation
    obsp_key
        Key in `.obsp` for the connectivity matrix to use
    normalize_with_obs_counts
        Whether to normalize layer before imputing

    Returns
    -------
    `adata.layers['imputed_<layer>']` for each layer in layers
    """
    impute_strength = float(impute_strength)
    ArgAssert(impute_strength < 1, 'Impute strength has to be in range (0.0, 1.0)')

    A: csr_matrix = adata.obsp[obsp_key].copy()
    #A.data = 1 / A.data
    if density_normalize:
        q = np.asarray(A.sum(axis=0))
        Q = spdiags(1.0 / q, 0, A.shape[0], A.shape[0])
        A = Q @ A @ Q
    else:
        A.data = A.data / A.data.max()

    if normalize_with_obs_counts:
        ls = [(adata.layers[k].astype(np.float32)) for k in layers]
        ls = [1000 * (diags(1 / l.sum(axis=1).A1) @ l).tocsr() for l in ls]
    else:
        ls = [adata.layers[k].tocsr() for k in layers]

    for i, k in enumerate(layers):
        X_imputed = (A @ ls[i]).astype(np.float32)
        adata.layers[f'imputed_{k}'] = X_imputed*impute_strength+adata.layers[k].astype(np.float32)*(1-impute_strength)
