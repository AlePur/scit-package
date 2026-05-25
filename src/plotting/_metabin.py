from anndata import AnnData
from matplotlib.pyplot import cm
from matplotlib.figure import Figure
import numpy as np

from ._helper import MplWrap, ColorUtil, NumColorUtil
from .._utils import ArgAssert
from .._logging import logging

def heatmap(
        adata: AnnData
) -> None:
    pass

def metabin(
        adata: AnnData,
        features: str | None | list[str] = None,
        *,
        size: float = 3,
        fsize: float = 50,
        show: bool = True
) -> tuple[dict, None|Figure]:
    """
    Plot metabin embedding, highlighting features (optional)

    Parameters
    ----------
    adata
    features
        Features to reveal on the embedding
    size
        Size in pixels of one bin plotted
    fsize
        Size of features
    show
        Show plot

    Returns
    -------
    Tuple with following two elements:

    dict
        Dictionary of feature names and corresponding metabin
    :class:`Figure` 
        If show == False
    """
    ArgAssert(('metabin' in adata.var.keys()) and ('X_embedding' in adata.varm.keys()), "Metabin memebership has to exist in .var, together with embedding in .varm. Please run toolkit.tl.add_metabin_metadata")
    plw = MplWrap(show)

    col_util = ColorUtil(adata,'', explicit_series=adata.var['metabin'])
    col_arr = col_util.get_col_arr()

    assert isinstance(col_util, NumColorUtil)

    _cm = cm.get_cmap('viridis')
    vmin, vmax = (col_arr.min(), col_arr.max()) #type:ignore
    X = adata.varm['X_embedding']

    sc = plw.ax.scatter(X[:,0], X[:,1],c=col_arr,s=size,vmin=vmin, vmax=vmax, cmap=_cm)
    cbar = plw.fig.colorbar(sc, ax=plw.ax)
    cbar.solids.set(alpha=1) #type:ignore
    plw.arrowed_spines()

    d = {}
    if features is not None:
        if not isinstance(features, list):
            features = [features]
        _sb = adata[:, np.isin(adata.var.index,features)]
        d = dict(zip(_sb.var.index.tolist(), _sb.var['metabin'].tolist()))
        plw.ax.scatter(_sb.varm['X_embedding'][:,0],_sb.varm['X_embedding'][:,1],c='red',s=fsize)

    return d, plw.show()