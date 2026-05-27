from anndata import AnnData
import anndata as ad

from scipy.sparse import csr_matrix
from ._metadata import add_metadata
from .._utils import ArgAssert
from ..tools._layers import LayerConfig

def transpose_anndata(
        adata: AnnData,
        layers: list[LayerConfig],
        n_features: int,
        *,
        copy_var: bool = True
) -> AnnData:
    """
    Transpose an AnnData object, creating a new AnnData where observations become variables.

    Parameters
    ----------
    adata
        Annotated data matrix to transpose
    layers
        Layer configurations specifying which layers to transpose into the new data
    n_features
        Number of features (rows) in the transposed matrix
    copy_var
        If True, copy ``adata.var`` into ``tdata.obs``

    Returns
    -------
    AnnData
        Transposed AnnData object
    """
    X = csr_matrix((n_features, adata.shape[0]))
    tdata = ad.AnnData(X)
    for lc in layers:
        _X = lc.transform(adata, raw_view=True)
        tdata.layers[lc.layer] = _X.T

    if copy_var:
        tdata.obs = adata.var.copy()
    tdata.var = adata.obs.copy()
    #add_metadata(tdata, var_loc_metadata=False)

    return tdata
