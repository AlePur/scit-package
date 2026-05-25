from anndata import AnnData
from typing import Any, Literal
import matplotlib.pyplot as plt
import numpy as np
import re

from ._helper import MplWrap
from .._utils import ArgAssert
from .._logging import logging
from ..tools._layers import LayerConfig

_rt = "Please run toolkit.tl.add_metadata"


def _check_lc(adata: AnnData) -> None:
    ArgAssert(
        len(list(adata.layers.keys())) != 0,
        "This anndata has no layers. Please fix this by using X_to_layer or stack_adata"
    )


# def default_style() -> None:
#    import matplotlib.pyplot as plt
#    plt.style.use("seaborn-v0_8-dark")

def _histogram(
        _data: Any,
        log_yscale: bool = True,
        title: str = '',
        xlabel: str = '',
        bins: int = 30,
        *,
        label_exp: bool = False,
        xminmax: tuple[float, float] | None = None
) -> tuple[MplWrap, Any]:
    plw = MplWrap(True)

    _, b, _ = plw.ax.hist(_data, bins)
    plw.ax.set_title(title)
    plw.ax.set_xlabel(xlabel)
    if log_yscale:
        plw.ax.set_yscale('log')
    if xminmax is not None:
        plw.ax.set_xlim(left=xminmax[0], right=xminmax[1])
    if label_exp:
        xticks = plw.ax.get_xticklabels()
        for i in range(len(xticks)):
            xticks[i]._text = '{0:.1f}'.format((np.e ** float(re.sub('−', '-', xticks[i]._text))) - 1)
        plw.ax.set_xticklabels(xticks)
    plw.despine()
    return plw, b


def degree_histogram(
        adata: AnnData,
        obsp_key: str,
        *,
        log_scale: bool = False,
        xminmax: tuple[float, float] | None = None
) -> None:
    degree = adata.obsp[obsp_key].getnnz(axis=1)

    _histogram(
        degree,
        log_scale,
        f'Distribution of cell neighbor-graph degree',
        'Degree',
        xminmax=xminmax
    )[0].show()

_htype = Literal['cell', 'feature']

def counts_histogram(
        adata: AnnData,
        log_counts: bool = True,
        log_y_scale: bool = False,
        hist_type: _htype = 'cell',
        *,
        label_exp: bool = False,
        xminmax: tuple[float, float] | None = None
) -> None:
    """
    Plot cell counts histogram in all layers, i.e. fragments/reads per cell.

    Parameters
    ----------
    adata
    log_y_scale
        Whether y-axis should be log scale
    """
    _check_lc(adata)
    if hist_type == 'cell':
        _obs = adata.obs
    elif hist_type == 'feature':
        _obs = adata.var
    else:
        raise ValueError()

    for k in _obs.keys():
        if k[-12:] != 'total_counts':
            continue

        _histogram(
            np.log1p(_obs[k]) if log_counts else _obs[k],
            log_y_scale,
            f'Distribution of {hist_type} total counts ({k})',
            f'ln1p({hist_type} total counts)',
            label_exp=label_exp,
            xminmax=xminmax
        )[0].show()


def layer_config_histogram(
        adata: AnnData,
        lc: LayerConfig,
        *,
        threshold: float | None = None,
        total_counts: bool = False,
        after_binarization: bool = False,
        log_xscale: bool = False,
        log_scale: bool = False,
        ignore_zeroes: bool = False,
        bins: int | None = None,
        xminmax: tuple[float, float] | None = None
) -> None:
    """
    Plot the sparse matrix data (or total feature counts when total_counts=True) of a layer config
    executed on given anndata. This function can be useful when deciding the threshold for active
    signal, since it plots the frequency of signal data.

    Parameters
    ----------
    adata
    lc
        LayerConfig
    threshold
        If this is None, use threshold set in layer config
    total_counts
        If this is True, plot frequency of feature signal not matrix data signal
    after_binarization
        Whether to binarize the counts and use the result for histogram
    log_xscale
        Histogram y-axis scale
    bins
        Number of bins
    xminmax
        xlim for plot

    """
    histarg = {
        "xminmax": xminmax,
    }
    if bins is not None:
        histarg["bins"] = bins
    m = lc.transform(adata)

    def passing_info(data, _threshold, total):
        passing = (data > _threshold).sum()
        fs = '{0:.2f}'

        logging.info(
            f"Passing entries (> {_threshold}): "
            f"{passing} ({fs.format(100 * passing / total)}%)"
        )

    if total_counts is False:
        ArgAssert(after_binarization is False, "after_binarization cannot be True when total_counts is False")

        if threshold is None:
            threshold = lc.get_threshold()

        passing_info(m.data, threshold, m.nnz)
        data = m.data.copy()
        if ignore_zeroes:
            data = data[data != 0]

        plw, _ = _histogram(
            np.log(data) if log_xscale else data,
            log_scale,
            f'Distribution of transformed {lc.layer} counts',
            'Transformed counts',
            **histarg
        )
        ymin, ymax = plw.ax.get_ylim()
        plw.ax.vlines(threshold, ymin, ymax, colors='r')
    else:
        if after_binarization:
            data = m.astype(np.bool_).sum(axis=0).A1
        else:
            data = m.sum(axis=0).A1

        if ignore_zeroes:
            data = data[data != 0]
        plw, b = _histogram(
            np.log(data) if log_xscale else data,
            log_scale,
            f'Distribution of per-feature {lc.layer} counts{"" if not after_binarization else " (binarized)"}',
            'Transformed total counts',
            **histarg
        )

        if after_binarization:
            if threshold is None:
                threshold = lc.get_threshold()
            m.data = m.data > threshold
            msum = m.sum(axis=0).A1
            if log_xscale:
                msum = np.log(msum)
                msum[msum == -np.inf] = 0
            plw.ax.hist(msum, bins=b, alpha=0.5, label='after thresholding')
            plw.fig.legend()
    plw.show()
