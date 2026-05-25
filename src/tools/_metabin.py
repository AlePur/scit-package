from anndata import AnnData
import numpy as np
from scipy.sparse import issparse, csr_matrix, coo_matrix

from .._utils import ArgAssert, _get_memberships
from .._logging import logging
from ._layers import LayerConfig

def membership_summary(
        adata: AnnData,
        layer_config: LayerConfig,
        obs_key: str,
        *,
        anti_summary: bool = False,
        normalize_per_class_obs_key: str | None = None,
        # normalize_keep_sum: bool = False,
        use_cached_mask: bool = True,
) -> dict:
    """
    Summarize data 
    """

    #    normalize_per_obs_category
    #    Minmax normalize values for every category separately
    X = layer_config.transform(adata, use_cached_mask=use_cached_mask).tocoo()
    #X_bool = coo_matrix((np.zeros_like(X.data, dtype=np.bool_), (X.row, X.col)), shape=X.shape)
    #X_bool_csr = X_bool.tocsr()
#
    #def crowcol(row, col, rowlen):
    #    return row.astype(np.int64) + col.astype(np.int64)*rowlen
    #def antic(rowcol, rowlen):
    #    # row, col
    #    return np.c_[rowcol % rowlen, rowcol // rowlen]

    d = {}
    cats, cat_names, uq_cats = _get_memberships(adata, obs_key)
    if normalize_per_class_obs_key is not None:
        if anti_summary:
            raise NotImplementedError("anti_summary can't be set as well as normalize_per_class_obs_key")
        class_cats, class_cat_names, class_uq_cats = _get_memberships(adata, normalize_per_class_obs_key)
        for uq in class_uq_cats:
            mask = class_cats == uq

            #masked_coo = X_bool_csr[mask].tocoo()
            #xmaskc = crowcol(masked_coo.row, masked_coo.col, X.shape[0])
            #ix = np.argsort(xmaskc)
            #data_mask = ix[np.searchsorted(xmaskc, crowcol(X_bool.row, X_bool.col, X.shape[0]), sorter=ix)]
            data_mask = mask[X.row]

            #if not normalize_keep_sum:
            x = 1e6
            #else:
            #    x = data_mask.shape[0]
            X.data[data_mask] = X.data[data_mask] * (x / X.data[data_mask].sum())

    X = X.tocsr()
    for uq in uq_cats:
        mask = cats == uq
        if anti_summary:
            mask = cats != uq
        
        d[cat_names[uq]] = X[mask]

    return d


def _membership_summary_on_array(
        adata: AnnData,
        obs_key: str,
        arr: np.ndarray,
        *,
        anti_summary: bool = False
) -> dict:
    """
    Summarize data 
    """

    d = {}
    cats, cat_names, uq_cats = _get_memberships(adata, obs_key)
    
    for uq in uq_cats:
        mask = cats == uq
        if anti_summary:
            mask = cats != uq
        
        d[cat_names[uq]] = arr[mask]

    return d


def summarize_data(
        adata: AnnData,
        layer_config: LayerConfig,
        obs_key: str | None = None,
        var_key: str | None = None,
        binary: bool = False,
        *,
        use_cached_mask: bool = False,
        normalize_with_bin_n: bool = True,
        metabin_std: bool = False,  # default True will break
) -> None:
    """
    Summarize data and save to either .obsm or .varm

    Parameters
    ----------
    adata
    layer_config
        Layer to summarize
    obs_key
        Group by categorical in obs
    var_key
        Group by categorical in var
    binary
        Binarize data
    use_cached_mask
        Use layer config cached_mask
    normalize_with_bin_n
        Normalize with amount of bins in metabin
    metabin_std
        Calculate standard deviation

    Returns
    -------
    .uns[sum_<layer for layer in layers>_<var_key>]

    """
    ArgAssert((obs_key is not None) or (var_key is not None), "either obs or var has to be used for summary")

    if metabin_std:
        ArgAssert(
            not ((obs_key is not None) and (var_key is not None)),
            "metabin_std cannot be True if summarization over both var and obs is done"
        )

    #    normalize_per_obs_category
    #    Minmax normalize values for every category separately
    ls = layer_config.transform(adata, use_cached_mask=use_cached_mask)

    if binary:
        ls.data = ls.data > layer_config.get_threshold()

    # Binary layers
    binary_ls = ls.copy()
    binary_ls.data = binary_ls.data.astype(np.bool_)
    #X_count_nonzero = None
    #X_count_nonzero_f = None

    if obs_key is not None:
        ArgAssert(adata.obs[obs_key].dtype == "category", "obs_key has to be categorical")
        oc = np.array(adata.obs[obs_key].cat.codes.values)
        oc_shape = adata.obs[obs_key].cat.categories.shape[0]
        X_after_obs = np.zeros((oc_shape, ls.shape[1]), dtype=np.float32)
        #X_count_nonzero = X_after_obs.copy()

        for i in range(oc.max() + 1):
            a = oc == i
            # set metabin bin expression
            X_after_obs[i] = (ls[a].sum(axis=0).A1.astype(np.float32) / a.sum())
            #X_count_nonzero[i] = (np.array(binary_ls[a].sum(axis=0), dtype=np.float32)[0] / a.sum())

    # X_after_obs shape: (max obs cat, adata.shape[1])
    else:
        X_after_obs = ls

    if var_key is not None:
        ArgAssert(adata.var[var_key].dtype == "category", "var_key has to be categorical")
        vc = np.array(adata.var[var_key].cat.codes.values)
        X_after_var = np.zeros((vc.max() + 1, X_after_obs[0].shape[0]), dtype=np.float32)
        X_std = X_after_var.copy()
        #X_count_nonzero_f = X_after_var.copy()

        for i in range(vc.max() + 1):
            a = vc == i
            s_or_d = X_after_obs[:, a]
            qnp = s_or_d.todense() if issparse(s_or_d) else s_or_d
            X_after_var[i] = (qnp.sum(axis=1)).ravel()
            if metabin_std:
                X_std[i] = (qnp.std(axis=1)).ravel()
            if normalize_with_bin_n:
                X_after_var[i] = X_after_var[i] / (a.sum())
            #if X_count_nonzero is not None:
            #    X_count_nonzero_f[i] = (np.array(X_count_nonzero[:, a].sum(axis=1)).ravel())
            #    if normalize_with_bin_n:
            #        X_count_nonzero_f[i] = X_count_nonzero_f[i] / (a.sum())
    else:
        X_after_var = X_after_obs
        # if (normalize_per_obs_category):
        # obs in in axis 0 right now

        # X_mdf=[((x - x.min()) / (x.max() - x.min())) for x in X_mdf]

    X_after_var = X_after_var.T
    if metabin_std:
        X_std = X_std.T
    # if (X_md is not None):
    #    X_mdf=[x.T for x in X_mdf]

    # if (total_sum is not None):
    #    X_after_var = [x * (total_sum / x.sum()) for x in X_after_var]
    targ = ''

    if (var_key is not None) and (obs_key is not None):
        _s = f'sum_{layer_config.layer}_{var_key}_{obs_key}'
        adata.uns[_s] = X_after_var

        targ = 'uns'
    else:
        targ = 'uns'
        if obs_key is not None:
            _s = f'sum_{layer_config.layer}_{obs_key}'
            adata.uns[_s] = X_after_var
        else:
            _s = f'sum_{layer_config.layer}_{var_key}'
            adata.uns[_s] = X_after_var
            if metabin_std:
                __s = f'std_{layer_config.layer}_{var_key}'
                adata.uns[__s] = X_std

        # if (X_md is not None):
        #    adata.uns[_s]['X_md'] = X_mdf[i]

    logging.info(f"Added {_s} to .{targ}")


def add_metabin_metadata(
        adata: AnnData,
        tdata: AnnData,
        obs_membership_key: str,
        embedding_key: str | None = None,
        add_embedding: bool = True,
        add_dendrogram: bool = True,
) -> None:
    """
    Add metabin metadata from a transposed anndata. Metabin membership is always added,\
    addition of embedding and dendrogram is controlled by boolean parameters.

    Parameters
    ----------
    adata
        Original adata
    tdata
        Transposed adata
    obs_membership_key
        Membership key (such as leiden or kmeans) in .obs
    embedding_key
        Key in tdata.obsm for embedding
    add_embedding
        Whether to add metabin embedding to adata
    add_dendrogram
        Whether to add dendrogram to adata

    Returns
    -------
    `adata.var['metabin']`
        Metabin membership
    `adata.varm['X_embedding']`
        Embedding
    `adata.uns['metabin_community_tree']`
        `community_tree` from tdata 

    """

    ArgAssert((adata.shape[1] == tdata.shape[0]), "Adata and tdata are not transposes of each other")
    ArgAssert((adata.var.index.to_numpy() == tdata.obs.index.to_numpy()).all(), "Adata var and tdata obs indices do not match")
    ArgAssert(obs_membership_key in tdata.obs.keys(), "obs_membership_key not in tdata.obs")
    adata.var['metabin'] = tdata.obs[obs_membership_key]

    if (add_embedding):
        logging.info("Adding embedding...")
        ArgAssert(embedding_key is not None, "embedding_key must not be None if add_embedding = True")
        ArgAssert(embedding_key in tdata.obsm.keys(), "embedding_key needs to be in tdata.obsm")
        adata.varm['X_embedding'] = tdata.obsm[embedding_key].copy()
    
    if (add_dendrogram):
        logging.info("Adding dendrogram...")
        ArgAssert("community_tree" in tdata.uns.keys(), "community_tree needs to be in tdata.uns")
        adata.uns['metabin_community_tree'] = tdata.uns["community_tree"].copy()
