from anndata import AnnData
import numpy as np
from pandas.api.types import is_numeric_dtype

from .._utils import ArgAssert
from .._logging import logging

def _verify_numeric(adata: 'AnnData', y_key: str) -> None:
    ArgAssert(y_key in adata.obs.keys(), "y_key not found in .obs")
    ykey = adata.obs[y_key].to_numpy()
    ArgAssert(is_numeric_dtype(ykey.dtype), "y_key is not numeric. Cannot use as y-coordinate")

def _verify_lsc(
        adata: 'AnnData', 
        y_key: str | None = None, # None for skip y key verification
    ) -> None:
    _mismatch = "mismatch between categorical community stored in .obs and community hierarchy. This probably happened because the categories in .obs were modified without running community_hierarchy. Try running toolkit.gr.community_hierarchy again"
    ArgAssert('community_tree' in adata.uns.keys(), "Please run toolkit.gr.community_hierarchy")
    
    if not (y_key is None):
        _verify_numeric(adata, y_key)
    o_k = adata.uns['community_tree']['obs_key']
    ArgAssert(o_k in adata.obs.keys(), _mismatch)

    membership = np.array(adata.obs[o_k].cat.codes.values)

    order = np.array(adata.uns['community_tree']['order'])
    ArgAssert(order.max() == membership.max(), _mismatch)

class LandscapeScaffold:

    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Scaffold on {self.obs_indices.shape[0]} cells"

    def __init__(self, memberships: list[np.ndarray], order: np.ndarray, obs_indices: list[str]) -> None:
        self.memberships = memberships #int
        self.order = order
        self.obs_indices = np.array(obs_indices) #string

    def _check_obs(self, adata: AnnData) -> None:
        if not (np.isin(self.obs_indices, adata.obs.index.tolist()).all()):
            raise ValueError("There are observation barcodes (indices) in the scaffold that don't exist in the anndata object")


def create_landscape_scaffold(
        adata: AnnData
) ->  LandscapeScaffold:
    """
    Get scaffold for creating landscape based on dendrogram-based order.

    Parameters
    ----------
    adata
        AnnData (presumably subset) to use for creating scaffold

    Returns
    -------
    :class:`LandscapeScaffold`
    """
    _verify_lsc(adata)

    order = np.array(adata.uns['community_tree']['order'])
    o_k = adata.uns['community_tree']['obs_key']
    membership = np.array(adata.obs[o_k].cat.codes.values)
    m_dict: list[list] = []
    #init
    for i in range(np.unique(membership).shape[0]):
        m_dict.append([])

    #index lists
    ix = adata.obs.index.tolist()
    for i in range(len(ix)):
        m_dict[membership[i]].append(i) 

    return LandscapeScaffold(
        [np.array(m) for m in m_dict], 
        order,
        ix
    )

