from sklearn.cluster import KMeans
from anndata import AnnData
import pandas as pd

def kmeans(
        adata: AnnData,
        n_clusters: int,
        embedding_key: str
) -> None:
    """
    Run K-means clustering using :class:`sklearn.cluster.KMeans`

    Parameters
    ----------
    adata
        Annotated data matrix with embedding in ``adata.obsm[embedding_key]``
    n_clusters
        Number of clusters
    embedding_key
        Key for embedding in .obsm

    Returns
    -------
    None
        Writes clustering result into ``adata.obs['kmeans']``
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(adata.obsm[embedding_key])
    adata.obs['kmeans'] = pd.Categorical(kmeans.labels_)