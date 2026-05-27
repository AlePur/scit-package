from anndata import AnnData
from scipy.stats import entropy
import polars as pl

from ..tools._layers import LayerConfig, layer_pick_features
from .._utils import _get_memberships

def shannon_entropy(
        adata: AnnData,
        lc: LayerConfig,
        class_obs_key: str,
        celltype_obs_key: str
) -> pl.DataFrame:
    """Compute Shannon entropy of feature activity across cell types within each class.

    For each class in ``class_obs_key``, calculates the entropy of the
    per-cell-type mean activity distribution for every feature.

    Parameters
    ----------
    adata
        AnnData object with categorical annotations in ``.obs``
    lc
        LayerConfig specifying which layer to transform
    class_obs_key
        Key in ``adata.obs`` for the class grouping
    celltype_obs_key
        Key in ``adata.obs`` for the cell-type grouping

    Returns
    -------
    Polars DataFrame with entropy and prevalence columns per class and feature
    """
    X = lc.transform(adata, use_cached_mask=True) # active
    class_cats, class_cat_names, class_uq_cats = _get_memberships(adata, class_obs_key)
    ct_cats, ct_cat_names, ct_uq_cats = _get_memberships(adata, celltype_obs_key)
    
    ds = []
    gs = []
    ls = []

    for i in range(len(class_uq_cats)):
        ds.append([])
        ls.append([])
        _mask = class_cats == class_uq_cats[i]
        gs.append(X[_mask].mean(axis=0).A1)
        for uq in ct_uq_cats:
            ds[i].append(X[_mask][ct_cats[_mask] == uq].mean(axis=0).A1)
            ls[i].append(ct_cat_names[uq])

    entropies = [entropy(ds[i], axis=0, base=2) for i in range(len(class_uq_cats))]

    datas = {}

    for i in range(len(class_uq_cats)):
        t = class_cat_names[class_uq_cats[i]]
        datas[f'entropy_{t}'] = entropies[i]
        datas[f'prev_{t}'] = gs[i]

    df=pl.DataFrame(
        {
            'name': lc.get_feature_names(),
            **datas
        }
    )

    return df