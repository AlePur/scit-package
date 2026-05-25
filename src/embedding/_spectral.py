import numpy as np
import scipy.sparse as sp
from typing import Any
from anndata import AnnData

from sklearn.preprocessing import normalize
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.sparse import diags, eye, hstack, csr_matrix
from .._utils import ArgAssert
from .._logging import logging
from ..tools._transforms import _idf, _l2norm, _mfrob

def _internal_spectral(
        X: csr_matrix,
        n_components: int,
        sd_weighted: bool
) -> tuple[np.ndarray, np.ndarray]:
    if (np.sum(X, axis=1) == 0).any():
        raise ValueError(f"Some cells have a total of 0 counts. Please exclude them to avoid division by zero")

    if (np.sum(X, axis=0) == 0).any():
        raise ValueError(f"Some features have a total of 0 counts. Please exclude them to avoid division by zero")

    # 2: Compute D <- diag(X(X^T 1)-1)
    d = (X @ (np.sum(X, axis=0)).A1) - 1.0

    outlier = d < 0.0001
    if outlier.any():
        raise ValueError(
            f"Some cells are isolated with no neighbors; this can be caused by not filtering the data. Please filter "
            f"cells based on e.g. counts or exclude the outliers manually "
            f"(outlier cell index: {np.arange(X.shape[0])[outlier]})"
        )

    D_inv = diags(1.0 / d)
    D_inv_sqrt = diags(1.0 / np.sqrt(d))

    # 3: Compute ~X <- D^(-1/2)X
    X_tilde = D_inv_sqrt @ X

    # 4: Define the vector transformation function F(v)
    def F(v: Any) -> Any:
        return X_tilde @ (v.T @ X_tilde).T - D_inv * v

    n = X.shape[0]  # type:ignore

    linear_op = LinearOperator(dtype=np.float64, shape=(n, n), matvec=F)

    # 5: Use Lanczos algorithm to get top k eigenvalues and eigenvectors
    eva, eve = eigsh(linear_op, k=n_components, which='LM')  # type:ignore
    eva[eva < 0] = 0

    if sd_weighted:
        eve = (eve * np.sqrt(eva))

    return eva, eve

def spectral(
        adata: AnnData,
        layer: str,
        n_components: int = 25,
        use_all_features: bool = False,
        *,
        weighted_by_sd: bool = True
) -> np.ndarray:
    """
    Spectral embedding. Adapted from SnapATAC2, DOI: `10.1038/s41592-023-02139-9`

    Parameters
    ----------
    layer
        Layer key for spectral
    n_components
        Number of components to generate.

    Returns
    -------
    `adata.obsm['X_spectral']`
        Embedding of dimensions (n, n_components)

    eigenvalues
        Eigenvalues that can be plotted with :func:`pl.explained_variance_ratio`
    """
    var_mask = np.full((adata.shape[1],), True)
    if not use_all_features:
        var_mask[adata.var['exclude']] = False
    ArgAssert(layer in adata.layers.keys(), "layer not found")

    X = _idf(adata.layers[layer][:, var_mask])  # type:ignore
    X_norm = _l2norm(X.tocsr())

    eigenvalues, eigenvectors = _internal_spectral(X_norm, n_components, weighted_by_sd)

    adata.obsm['X_spectral'] = np.flip(eigenvectors, axis=1)
    return np.flip(eigenvalues)

def multiview_spectral(
        adata: Any,
        layers: list[str],
        *,
        layer_weights: list[float] | None = None,
        n_components: int = 25,
        random_seed: int | None = None,
        use_all_features: bool = False,
        skip_idf: bool = False,
        weighted_by_sd: bool = True
) -> np.ndarray:
    """
    Multiview spectral embedding. Adapted from SnapATAC2, DOI: `10.1038/s41592-023-02139-9`

    Parameters
    ----------
    layer_weights
        List of floats with layer importances
    n_components
        Number of components to generate.
    random_seed
        Random seed for sampling data. If set to None, a seed is generated internally
    skip_idf
        Skip idf. This is not recommended!

    Returns
    -------
    `adata.obsm['X_multi_spectral']`
        Embedding of dimensions (n, n_components)

    eigenvalues
        Eigenvalues that can be plotted with :func:`pl.explained_variance_ratio`
    """

    var_mask = np.full((adata.shape[1],), True)
    if not use_all_features:
        var_mask[adata.var['exclude']] = False
    try:
        if skip_idf:
            X_list = [adata.layers[l][:,var_mask].tocsr() for l in layers]
        else:
            X_list = [_idf(adata.layers[l][:,var_mask].tocsr()) for l in layers]
    except Exception as e:
        logging.warning("Cannot find some of the layers")
        raise e

    if (layer_weights is None):
        lambdas = np.ones((len(layers),))
    else:
        lambdas = np.array(layer_weights)

    gen = np.random.default_rng(seed=random_seed)
    sample_k = int(float(adata.shape[0]) / 2.0)  # TODO: find reason for this

    # Step 1: Normalize each Xi
    ws = np.zeros((len(X_list)), dtype=np.float64)
    X_norm = [_l2norm(X) for X in X_list]

    for i in range(len(X_list)):
        X_sample = X_norm[i][gen.choice(adata.shape[0], size=sample_k, replace=False), :]
        ws[i] = lambdas[i] / _mfrob((X_sample @ X_sample.T) - eye(sample_k))

    X_normalized = [X_norm[i].multiply(np.sqrt(ws[i] / ws.sum())) for i in range(len(X_list))]

    # Step 2: Concatenate the matrices horizontally
    eigenvalues, eigenvectors = _internal_spectral(hstack(X_normalized).tocsr(), n_components, weighted_by_sd)

    adata.obsm['X_multi_spectral'] = np.flip(eigenvectors, axis=1)
    return np.flip(eigenvalues)
