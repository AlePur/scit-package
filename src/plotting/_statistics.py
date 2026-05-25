import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.stats as sts
from anndata import AnnData
from numpy.typing import NDArray
import numpy as np

from .int import scatter
from .._utils import ArgAssert
#from .._settings import settings
from ._helper import MplWrap


def volcano_plot(
        pval: np.ndarray,
        lfc: np.ndarray,
        *,
        min_pval: float = 0.05,
        min_lfc: float = 1,
        yminmax: tuple[float, float] = (-0.5, 10),
        xminmax: tuple[float, float] = (-3, 3),
        zero_to_min: float = True,
        show: bool = True,
        s: int = 1
):
    
    _pval = pval.copy()
    if zero_to_min:
        _pval[_pval == 0] = np.nan
        _pval[np.isnan(_pval)] = np.nanmin(_pval)
    is_sig = (_pval < min_pval) * np.abs(lfc) >= min_lfc
    f = scatter(
        np.c_[lfc, -np.log10(_pval)],
        xminmax=xminmax, show=False, alpha=1, s=s
    )
    f.axes[0].scatter(
        lfc[~is_sig], -np.log10(_pval[~is_sig]), s=s, c='tab:gray'
    )
    f.axes[0].set_ylim(*yminmax)
    kwarg = dict(
        transform=f.axes[0].transAxes,
        fontsize=10,
        verticalalignment='center',
        wrap=True
    )
    f.axes[0].text(
        0.05, 0.95,
        str(((lfc < 0) * is_sig).sum()) + ' down',
        **kwarg
    )
    f.axes[0].text(
        0.85, 0.95,
        str(((lfc > 0) * is_sig).sum()) + ' up',
        **kwarg
    )
    f.axes[0].set_ylabel('-log10(adjP)')
    f.axes[0].set_xlabel('log2FC')
    f.axes[0].hlines(-np.log10(min_pval), xminmax[0], xminmax[1], color='tab:red')
    f.axes[0].vlines(-min_lfc, yminmax[0], yminmax[1], color='tab:red')
    f.axes[0].vlines(min_lfc, yminmax[0], yminmax[1], color='tab:red')
    if show:
        plt.show()
        plt.close(f)
        return None
    return f

def explained_variance_ratio(
        explained_variance: NDArray,
        *,
        n_pcs: int | None = None,
        show: bool = True
) -> None | Figure:
    """
    Plot explained variance ratio from :func:`toolkit.tl.pca`

    Parameters
    ----------
    explained_variance
        Array of explained variance
    n_pcs
        Number of principal components to plot
    show
        Show plot

    """
    explained_variance = explained_variance[:n_pcs]

    plw = MplWrap(show)
    
    plw.ax.scatter(range(explained_variance.shape[0]), explained_variance)
    plw.despine()
    return plw.show()

def depth_corr(
        adata: AnnData,
        embedding_key: str,
        layer: str,
        show: bool = True
) -> None | Figure:
    """
    Calculate correlation between total counts and embedding dimensions

    Parameters
    ----------
    adata
    embedding_key
        Embedding in .obsm to use for calculating correlation
    layer
        Layer to use for calculating correlation
    show
        Show plot
    """
    ArgAssert(layer in adata.layers.keys(), "layer not in .layers")
    ArgAssert(embedding_key in adata.obsm.keys(), "embedding_key not in .obsm")
    ArgAssert(f'{layer}_total_counts' in adata.obs.columns, "please run toolkit.tl.add_metadata")

    pers=[]
    for s in range(adata.obsm[embedding_key].shape[1]):
        arr=adata.obsm[embedding_key][:,s]
        pers.append(sts.pearsonr(adata.obs[f'{layer}_total_counts'], arr))

    plw = MplWrap(show)
    plw.ax.scatter(range(adata.obsm[embedding_key].shape[1]), [p.statistic for p in pers])
    plw.despine()
    return plw.show()