from anndata import AnnData
import numpy as np
from typing import Any
from scipy.stats import gaussian_kde

from .._helper import MplWrap
from ..._utils import _get_memberships, ArgAssert

def density_estimate(data):
    density = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 100)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    return np.array(xs), np.array(density(xs))

def histogram(
        datas: np.ndarray | list[np.ndarray],
        labels: list[str] | None = None,
        *,
        bins: int = 50,
        alpha: float = 0.6,
        title: str | None = None,
        colors: list[str] | tuple[str] = ('tab:blue', 'tab:orange', 'tab:green'),
        xminmax: tuple | None = None,
        max_normalize: bool = False,
        kde: bool = False,
        kde_scalar: float = 1,
        show: bool = True
) -> None:
    """
    Create histogram.

    Parameters
    ----------
    datas
    labels
    bins
        Number of bins to use for histogram
    alpha
        Alpha for plotting
    title
        Title of plot
    colors
        Colors list for each data
    xminmax
        Min, max of x-axis
    kde
        Whether to plot Kernel Density Estimate on top
    kde_scalar
        Scale KDE by this amount

    """
    plw = MplWrap(show)
    if not isinstance(datas, list):
        datas = [datas]
    l = {}
    ArgAssert(len(colors) >= len(datas), 'please supply colors with parameter colors=')
    l['color'] = colors[0]
    if labels is not None:
        l['label'] = labels[0]
    if max_normalize:
        l['density'] = True

    _min = np.min([d.min() for d in datas])
    _max = np.max([d.max() for d in datas])
    b = (np.arange(bins)/bins) * (_max - _min) + _min
    for i in range(len(datas)):
        l['color'] = colors[i]
        if labels is not None:
            l['label'] = labels[i]
        plw.ax.hist(datas[i], bins=b, alpha=alpha, **l)
    if kde:

        for i in range(len(datas)):
            dx = density_estimate(datas[i])
            plw.ax.plot(dx[0], dx[1] * kde_scalar, color=colors[i])
    if title is not None:
        plw.ax.set_title(title)
    if labels is not None:
        plw.fig.legend()
    if xminmax is not None:
        plw.ax.set_xlim(left=xminmax[0], right=xminmax[1])
    plw.despine()
    return plw.show()
