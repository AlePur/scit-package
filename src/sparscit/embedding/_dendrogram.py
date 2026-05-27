import numpy as np
from anndata import AnnData
from ..advanced._landscape_config import _verify_lsc

def dendrogram(
        adata: AnnData,
        y_key: str,
        *,
        random_seed: int = 0
) -> None:
    """
    Create embedding based on dendrogram order.

    Writes a 2D embedding into ``adata.obsm['X_dendrogram']`` where the x-coordinate
    is derived from community tree ordering and the y-coordinate from the specified
    observation key.

    Parameters
    ----------
    adata
        Annotated data matrix with community tree information in ``adata.uns``
    y_key
        Key in .obs to use for y-coordinate of cell on embedding
    random_seed
        Random seed for jitter applied to x-coordinates

    Returns
    -------
    None
        Writes the result into ``adata.obsm['X_dendrogram']``
    """
    _verify_lsc(adata, y_key)
    ykey = adata.obs[y_key].to_numpy()
    order = np.array(adata.uns['community_tree']['order'])
    o_k = adata.uns['community_tree']['obs_key']
    membership = np.array(adata.obs[o_k].cat.codes.values)

    _inverse_o = np.arange(order.max()+1)
    _inverse_o[order] = np.arange(order.max()+1)

    gen = np.random.default_rng(seed=random_seed)
    X_new = np.zeros((adata.shape[0], 2))
    X_new[:,1] = (ykey - ykey.min()) / (ykey.max() - ykey.min())
    X_new[:,0] = _inverse_o[membership]
    X_new[:,0] += gen.random((adata.shape[0], ))*0.8
    X_new[:,0] /=  (order.max()+1)

    adata.obsm['X_dendrogram'] = X_new * 100
