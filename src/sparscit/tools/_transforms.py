import numpy as np
from scipy.sparse import csr_matrix, diags, eye

def _mfrob(X: csr_matrix) -> np.float32:
    """Compute the Frobenius norm of a sparse matrix."""
    return np.sqrt(np.square(X.data).sum())

def _l2norm(X: csr_matrix) -> csr_matrix:
    """L2-normalize each row of a sparse matrix."""
    X2 = X.copy()
    X2.data = np.square(X2.data)
    nr = 1.0 / np.sqrt(X2.sum(axis=1).A1)
    # nr[nr == 0] = 1
    return diags(nr) @ X

def _idf(
        X: csr_matrix,
        allow_zeros: bool = False
) -> csr_matrix:
    """
    Apply inverse-document-frequency weighting to a sparse matrix.

    Each column is scaled by ``log(n_rows / doc_freq)``, where ``doc_freq`` is the
    number of non-zero entries in that column.

    Parameters
    ----------
    X
        Sparse count matrix of shape (n_obs, n_features)
    allow_zeros
        If False (default), raise an error when any column has zero non-zero entries.
        If True, replace zero frequencies with 1 to avoid division by zero.

    Returns
    -------
    :class:`scipy.sparse.csr_matrix`
        IDF-weighted sparse matrix
    """
    # If this crashes, probably modify .data directly
    frequency = X.getnnz(axis=0)
    if not allow_zeros:
        if (frequency == 0).any():
            raise ValueError(
                "There are features with zero total counts. "
                "Regenerate metadata and exclude them using toolkit.tl.filter"
            )
    else:
        frequency[frequency == 0] = 1
    return X @ diags(np.log(X.shape[0] / frequency))