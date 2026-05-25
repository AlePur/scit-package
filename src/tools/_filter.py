from anndata import AnnData
import numpy as np

from ._metadata import add_metadata
from .._utils import ArgAssert
from .._logging import logging
from typing import Any


def filter(
        adata: AnnData,
        layers: list[str],
        min_obs_counts: list[int | complex] | None = None,
        min_var_counts: list[int | complex] | None = None,
        *,
        max_obs_counts: list[int | complex] | None = None,
        max_var_counts: list[int | complex] | None = None,
        return_purged: bool = False,
        filter_iteratively: bool = True
) -> AnnData | None:
    """
    Parameters
    ----------
    adata
    layers
        Which layers to consider
    min_obs_counts, max_obs_counts, min_var_counts, max_var_counts
        If `None`, don't create filter based on min/max counts. List with one integer for each layer. If a complex number between 0 and 100 is provided, this is the percentile cutoff.
    filter_iteratively
        Iterate until filters thresholds are enforced
    return_purged
        Whether cells not passing the filter are removed from the anndata. The new filtered anndata is returned

    Returns
    -------
    `adata.obs['exclude']`, `adata.var['exclude']` and filtered adata if `return_purged == True`
    """
    obs_mask = np.full((adata.shape[0],), True)
    var_mask = np.full((adata.shape[1],), True)

    def edit_mask(m: np.ndarray, counts: np.ndarray, th: tuple[float, ...], k: str) -> np.ndarray:

        if th[0] is not None:
            m[counts <= th[0]] = False
        if th[1] is not None:
            m[counts >= th[1]] = False
        return m

    _obs_counts = [adata.layers[l].sum(axis=1).A1 for l in layers]
    _var_counts = [adata.layers[l].sum(axis=0).A1 for l in layers]

    def check(fc: list | None, counts: list[np.ndarray]) -> None | list:
        if fc is None:
            return [None] * len(layers)
        _le = "min/max counts argument has to be a list of same length as the number of layers in adata"
        ArgAssert(isinstance(fc, list), _le)
        ArgAssert(len(fc) == len(layers), _le)
        for i, _lt in enumerate(fc):
            if isinstance(_lt, complex):
                q = float(_lt.imag)
                th = np.percentile(counts[i], q)
                fc[i] = th
                logging.info(
                    f'The {q}th percentile of {"obs" if counts[i].shape[0] == adata.shape[0] else "var"}'
                    f' total counts is {th}'
                )
        return fc

    min_obs_counts = check(min_obs_counts, _obs_counts)
    max_obs_counts = check(max_obs_counts, _obs_counts)
    min_var_counts = check(min_var_counts, _var_counts)
    max_var_counts = check(max_var_counts, _var_counts)

    changed = True
    while changed:
        changed = False
        _obs_counts = [adata[obs_mask][:, var_mask].layers[l].sum(axis=1).A1 for l in layers]
        _m = np.full((obs_mask.sum(),), True)
        for i, k in enumerate(layers):
            _m *= edit_mask(obs_mask[obs_mask], _obs_counts[i], (min_obs_counts[i], max_obs_counts[i]), k)
        obs_mask[obs_mask] = _m
        changed += (~_m).sum(dtype=np.bool_)
        _m = np.full((var_mask.sum(),), True)
        _var_counts = [adata[obs_mask][:, var_mask].layers[l].sum(axis=0).A1 for l in layers]
        for i, k in enumerate(layers):
            _m *= edit_mask(var_mask[var_mask], _var_counts[i], (min_var_counts[i], max_var_counts[i]), k)
        var_mask[var_mask] = _m
        changed += (~_m).sum(dtype=np.bool_)

        if not filter_iteratively:
            changed = False

    adata.obs['exclude'] = ~obs_mask
    adata.var['exclude'] = ~var_mask

    if return_purged:
        adata = adata[obs_mask, var_mask].copy()
        return adata
    else:
        logging.info(
            "Adata not modified. If you want to remove excluded cells and bins, use return_purged = True "
            "or remove them manually"
        )
    return None

def remove_pc(
        adata: AnnData,
        embedding_key: str,
        index: int
) -> None:
    """
    Remove PC from embedding. This is useful when a PC is highly correlated with total counts

    Parameters
    ----------
    adata
    embedding_key
        Embedding in .obsm to modify
    index
        Index of PC to remove
    """

    ArgAssert(embedding_key in adata.obsm.keys(), "embedding_key not in .obsm")
    ArgAssert(adata.obsm[embedding_key].shape[0] > index, "index too large")
    adata.obsm[embedding_key]=np.delete(adata.obsm[embedding_key],index,axis=1)