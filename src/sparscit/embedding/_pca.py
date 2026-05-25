
from anndata import AnnData
from sklearn.decomposition import PCA
from numpy.typing import NDArray

from .._utils import ArgAssert
#from .._settings import settings
from .._logging import warn_overwrite


def pca(
    adata: AnnData,
    embedding_key: str | None = None,
    *,
    layer_key: str | None = None,
    n_components: int = 30
) -> NDArray:
    """\
    PCA using :class:`sklearn.decomposition.PCA`

    Parameters
    ----------
    adata
        AnnData object
    embedding_key
        Key of embedding in `.obsm` to transform
    layer_key
        Alternatively, a layer can be transformed if this key is provided
    n_components
        Number of principal components to keep

    Returns
    -------
    `adata.obsm['X_pca']`
        PCA embedding

    variance_ratio
        Variance ratio which can be plotted with :func:`sparscit.pl.explained_variance_ratio`
    """

    warn_overwrite('X_pca', adata.obsm)

    if (embedding_key is not None):
        ArgAssert(embedding_key in adata.obsm.keys(), "embedding_key not found in .obsm")
        X = adata.obsm[embedding_key]
    elif (layer_key is not None):
        ArgAssert(layer_key in adata.layers.keys(), "layer_key not found in .layers")
        X = adata.layers[layer_key]
    else:
        raise ValueError("Either embedding_key or layer_key needs to be non-None")

    pca = PCA(
        n_components=n_components,
        svd_solver='arpack'
    )
    X_new = pca.fit_transform(X)
    adata.obsm['X_pca'] = X_new
    return pca.explained_variance_ratio_
