from anndata import AnnData
from functools import reduce
import numpy as np
from scipy.sparse import csr_matrix
import anndata as ad

from .._utils import ArgAssert
from .._logging import logging

def set_missing_to_zero(adata, feature_list):
    missing=feature_list[~np.isin(feature_list, adata.var_names)]
    print(f"{len(missing)} features missing.")

    n_adata = ad.AnnData(csr_matrix((adata.shape[0], len(missing))))
    n_adata.var_names = missing
    n_adata.obs = adata.obs.copy()
    n_adata.layers['rna'] = csr_matrix((adata.shape[0], len(missing)), dtype=np.uint32)

    return ad.concat((adata, n_adata), axis='var', merge='first')

def X_to_layer(
        adata: AnnData,
        layer_name: str
) -> None:
    """
    Move X to .layers. Note! This replaces X with an empty csr matrix
    """
    if (layer_name in adata.layers.keys()):
        logging.warning("Overwriting layer...")
    try:
        x = adata.X.tocsr()  #type:ignore
    except:
        x = adata.X.copy()  #type:ignore
    adata.layers[layer_name] = x
    adata.X = csr_matrix((np.array([]), ([], [])), shape=adata.shape)

def stack_adata(
        adatas: list[AnnData],
        layer_names: list[str],
        keep: int = 0
) -> AnnData:
    """
    Add adata.X from each adata in adatas as a layer

    Parameters
    ----------
    adatas
    layer_names
        List of layer that adatas will form
    keep
        Index of the adata that will be used as base for adding layers

    Returns
    -------
    :class:`AnnData`
    """

    ArgAssert(len(adatas) == len(layer_names), "Length of layer names and adatas needs to be equal")

    common_obs = reduce(np.intersect1d, tuple([np.array(a.obs.index.tolist()) for a in adatas]))
    logging.info(f"found {common_obs.shape[0]} obs shared between all adatas...")

    common_var = reduce(np.intersect1d, tuple([np.array(a.var.index.tolist()) for a in adatas]))
    logging.info(f"found {common_var.shape[0]} var shared between all adatas...")

    adatas = [ada[common_obs,common_var] for ada in adatas]
    _adata = adatas[keep]

    for i in range(len(adatas)):
        if (adatas[i].X.data.shape[0] == 0):
            raise ValueError("All adatas in adata must have data in adata.X. Please move it there")
        _adata.layers[layer_names[i]] = adatas[i].X.copy()

    return _adata