from anndata import AnnData
import numpy as np
from typing import Any, Literal
import scipy.stats as sts

from matplotlib.figure import Figure
from .._helper import MplWrap
from ..._utils import _get_memberships, ArgAssert

_corr_t = Literal['spearman', 'pearson']

def scatter(
        datas: np.ndarray | list[np.ndarray],
        labels: list[str] | None = None,
        *,
        alpha: float = 1,
        title: str | None = None,
        colors: list[str] | tuple[str] = ('tab:blue', 'tab:orange', 'tab:green'),
        c_map: list[np.ndarray] | None = None,
        xminmax: tuple | None = None,
        yminmax: tuple | None = None,
        s: int = 3,
        xlabel: str | None = None,
        ylabel: str | None = None,
        corr: _corr_t | None = None,
        corr_use_minmax: bool = False,
        show: bool = True
) -> None | Figure:
    plw = MplWrap(show)
    if not isinstance(datas, list):
        datas = [datas]
    ArgAssert(len(colors) >= len(datas), 'please supply colors with parameter colors=')

    l = {}
    for i in range(len(datas)):
        l['color'] = colors[i]
        if labels is not None:
            l['label'] = labels[i]
        if c_map is not None:
            del l['color']
            l['c'] = c_map[i]
        plw.ax.scatter(datas[i][:,0], datas[i][:,1], s=s, alpha=alpha, **l)

    if title is not None:
        plw.ax.set_title(title)
    if labels is not None:
        plw.fig.legend()
    if xminmax is not None:
        plw.ax.set_xlim(xminmax[0], xminmax[1])
    if yminmax is not None:
        plw.ax.set_ylim(yminmax[0], yminmax[1])
    if xlabel is not None:
        plw.ax.set_xlabel(xlabel)
    if ylabel is not None:
        plw.ax.set_ylabel(ylabel)

    if corr is not None:
        for i in range(len(datas)):
            _x = datas[i][:,0]
            _y = datas[i][:,1]
            if corr_use_minmax:
                m = (_x > xminmax[0]) * (_x < xminmax[1]) * (_y > yminmax[0]) * (_y < yminmax[1])
                _x = _x[m]
                _y = _y[m]

            if corr == 'pearson':
                print(f"data list {i} correlation result: {sts.pearsonr(_x, _y)}")
            elif corr == 'spearman':
                print(f"data list {i} correlation result: {sts.spearmanr(_x, _y)}")
            else:
                raise ValueError(corr)
    plw.despine()
    return plw.show()
