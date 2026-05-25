
from anndata import AnnData
from .._utils import ArgAssert
from .._logging import warn_overwrite

def umap(
    adata: AnnData,
    embedding_key: str,
    *,
    n_neighbors: int = 15,
    umap_kwargs: dict = {}
) -> None:
    """\
    UMAP 2D embedding

    Parameters
    ----------
    embedding_key
        Which embedding to transform in .obsm
    n_neighbors
        Number of neighbors to consider for calculating UMAP

    Returns
    -------
    `adata.obsm["X_umap"]`
        UMAP embedding
     
    """

    from umap import UMAP
    warn_overwrite('X_umap', adata.obsm)
    ArgAssert(embedding_key in adata.obsm.keys(), "embedding_key not found in .obsm")

    X = adata.obsm[embedding_key]#[:use_n_components]

    X_new = UMAP(n_neighbors, **umap_kwargs).fit_transform(X)
    adata.obsm['X_umap'] = X_new
