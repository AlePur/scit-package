import numpy as np
from scipy.sparse import csr_matrix, diags, eye

def _mfrob(X: csr_matrix) -> np.float32:
    return np.sqrt(np.square(X.data).sum())

def _l2norm(X: csr_matrix) -> csr_matrix:
    X2 = X.copy()
    X2.data = np.square(X2.data)
    nr = 1.0 / np.sqrt(X2.sum(axis=1).A1)
    # nr[nr == 0] = 1
    return diags(nr) @ X

def _idf(
        X: csr_matrix,
        allow_zeros: bool = False
) -> csr_matrix:
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