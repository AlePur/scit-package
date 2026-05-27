from typing import Literal, Union, Annotated, Any, Tuple, Callable
import scipy.sparse as sps
from anndata import AnnData
import numpy as np


def save_sparse(_X, target: str):
    """Save a sparse matrix to disk in numpy .npz format."""
    X = _X.tocsr()
    np.savez(
        target, data=X.data, indices=X.indices,
        indptr=X.indptr, shape=X.shape
        )


def load_sparse(path: str):
    """Load a sparse matrix from a .npz file saved by :func:`save_sparse`."""
    loader = np.load(path)
    X = sps.csr_matrix(
        (loader['data'], loader['indices'], loader['indptr']),
        shape=loader['shape']
        )
    return X

def _csr_mean(_X: sps.csr_matrix | Any, axis: int, ignore_zero: bool = False, warn_not_csr: bool = True) -> np.ndarray:
    """Compute the mean of a CSR sparse matrix along an axis, optionally ignoring zeros."""
    if not isinstance(_X, sps.csr_matrix):
        ArgAssert(not warn_not_csr, 'Input not CSR matrix')
        return _X.mean()
    d = _X.getnnz(axis=axis)
    if ignore_zero:
        d[d == 0] = 1
    return _X.sum(axis=axis).A1 / d

def _csr_std(_X: sps.csr_matrix | Any, axis: int, ignore_zero: bool = False, warn_not_csr: bool = True, ddof: int = 0, return_mean: bool = False) -> np.ndarray:
    """
    Calculate standard deviation of a CSR matrix along a specified axis.
    """
    if not isinstance(_X, sps.csr_matrix):
        ArgAssert(not warn_not_csr, 'Input not CSR matrix')
        return _X.std(axis=axis, ddof=ddof)
    
    assert axis in [0,1], "Axis you provided is not supported"
    # Calculate mean
    mean = _csr_mean(_X, axis=axis, ignore_zero=ignore_zero, warn_not_csr=False)
    
    d = _X.getnnz(axis=axis)
    if ignore_zero:
        d[d == 0] = 1
    
    # Calculate variance: E[X^2] - E[X]^2
    # For sparse matrices, we need to handle this carefully
    _X_squared = _X.copy()
    _X_squared.data = np.square(_X_squared.data)
    mean_of_squares = _X_squared.sum(axis=axis).A1 / d
    
    variance = mean_of_squares - np.square(mean)
    
    # Adjust for degrees of freedom
    if ddof != 0 and not ignore_zero:
        n = _X.shape[1 - axis]
        variance = variance * n / (n - ddof)
    
    # Handle numerical errors that might make variance slightly negative
    variance = np.maximum(variance, 0)
    
    if return_mean:
        return np.sqrt(variance), mean
    return np.sqrt(variance)

def _double_lambda(_in: list[tuple[Any, Any]], lm: Callable) -> tuple[Any, Any]:
    """Apply a function separately to the first and second elements of a list of tuples."""
    first_in = [i[0] for i in _in]
    second_in = [i[1] for i in _in]
    return lm(*first_in), lm(*second_in)

def _most_frequent_uint(
        a: np.ndarray
) -> np.ndarray:
    """Return the most frequent value in an array of unsigned integers."""
    counts = np.bincount(a)
    return np.argmax(counts)

def _get_membership(
        adata: 'AnnData',
        community_key: str
) -> np.ndarray:
    """
    Get community membership from .obs
    """
    ArgAssert(community_key in adata.obs.keys(), "community_key not in .obs")
    ArgAssert(
        adata.obs[community_key].dtype.name == "category",
        "the community membership column in .obs needs to be Categorical. Please convert the column manually using pd.Categorical"
    )
    return adata.obs[community_key].cat.codes.values

def _get_memberships(
        adata: AnnData,
        obs_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get integer codes, category names, and unique codes for a categorical obs column.

    Parameters
    ----------
    adata
        Annotated data matrix
    obs_key
        Key in ``adata.obs`` (must be categorical)

    Returns
    -------
    cats : np.ndarray
        Integer codes for each observation
    cat_names : np.ndarray
        Category name for each code
    uq_cats : np.ndarray
        Sorted unique integer codes
    """
    ArgAssert(adata.obs[obs_key].dtype == "category", "obs_key has to be categorical")
    cats = adata.obs[obs_key].cat.codes.to_numpy()
    cat_names = adata.obs[obs_key].cat.categories.to_numpy()
    uq_cats = np.unique(cats)
    return cats, cat_names, uq_cats

def _check_obs_uq(
        adata: 'AnnData',
        fatal: bool = True
) -> None:
    """Verify that observation names in ``adata.obs.index`` are unique."""
    i, q = np.unique(adata.obs.index, return_counts=True)
    c = (i[q > 1]).shape[0]
    if c > 1:
        raise ValueError(
            "The obs names are not unique. Please make them unique manually or use adata.obs_names_make_unique()"
            )


def _map_dict(
        _dict: dict,
        l: Callable
) -> dict:
    """Apply a function to the values of a dictionary, preserving keys."""
    v = list(_dict.values())
    k = list(_dict.keys())
    mapped = l(v)
    return dict((_k, mapped[i]) for i, _k in enumerate(k))


def ArgAssert(
        condition: bool | np.bool_,
        msg: str
) -> None:
    """Raise ValueError with ``msg`` if ``condition`` is False."""
    if condition == False:
        raise ValueError(msg)


# Metric stuff copied from scanpy implementation

_MetricSparseCapable = Literal[
    "cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"
]
_MetricScipySpatial = Literal[
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "kulsinski",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
Metric = Union[_MetricSparseCapable, _MetricScipySpatial]


def symmetric_i(_max: int) -> np.ndarray:
    """Generate all unique pairs (i, j) with i < j for j in range(_max)."""
    a = []
    for i in range(_max):
        for j in range(i + 1, _max):
            a.append([i, j])
    return np.array(a)
