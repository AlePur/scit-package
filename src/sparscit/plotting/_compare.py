from anndata import AnnData
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ._helper import ColorUtil, CategoricalColorUtil, MplWrap
from .._utils import ArgAssert, _get_memberships
from ._bar import barplot, CWheel
from .._settings import settings
from .._logging import logging
from ..tools._statistics import Markers


def frequency_pie(
        data: np.ndarray | None = None,
        labelmap: dict | None = None,
        *,
        title: str | None = None,
        colors: list[str] | None = None,
        precomputed_freq: dict | None = None,
        color_map: dict | None = None,
        filter_min: int = -1,
        show_frac: bool = False,
        text_size: float = 15,
        ax: None | Axes = None
) -> None:
    """
    Pie plot for frequencies of categorical data

    Parameters
    ----------
    data
        Integer array
    labelmap
        Dictionary mapping integers to labels
    title
        Title of plot
    colors
        Colors used on pie
    precomputed_freq
        If this is passed, data parameter is ignored
    filter_min
        Minimum number of frequency of categorical class for it to be visible
    ax
        Pass axes to plot on

    Returns
    -------

    """
    if precomputed_freq:
        freq = precomputed_freq
    else:
        uq, c = np.unique(data, return_counts=True)
        uq = list(uq)
        c = list(c)
        if labelmap is not None:
            uq = [labelmap.get(u) for u in uq]
        freq = dict(zip(uq, c))

    tot = np.array(list(freq.values())).sum()
    ks = list(freq.keys())

    plw = None
    if ax is None:
        plw = MplWrap(True)
        ax = plw.ax

    lb = [a if freq[a] > filter_min else '' for a in freq.keys()]
    kw = {}
    if colors is not None:
        kw['colors'] = colors
    if color_map is not None:
        kw['colors'] = [color_map.get(l, 'tab:gray') for l in lb]
    if text_size is not None:
        kw['textprops']={'fontsize': text_size}
    ax.pie(
        freq.values(),
        labels=lb,
        autopct='%1.0f%%',
        **kw
    )
    if title is not None:
        ax.title.set_text(title)

def top_markers(
        markers: Markers,
) -> None:
    """
    Plot top markers

    Parameters
    ----------
    markers
        :class:`Markers`

    """
    ArgAssert(markers.top_markers is not None, "Please run filter_top_markers first")

    for group, df in markers.top_markers:
        if df is not None:
            barplot(
                df['nlog_p'].to_numpy(),
                df['name'].to_numpy(),
                df['pval'].to_numpy() < markers.alpha,
                ('tab:blue', 'tab:gray'),
                title=group
            )
        else:
            logging.warning('Nothing to show')

def compare_categorical(
        adata: AnnData,
        obs_base: str,
        obs_key: str,
        *,
        filter_min: float = 0.05
) -> None | Figure:
    """
    Compare categoricals using pie charts

    Parameters
    ----------
    adata
    obs_base
        Key to base comparison on
    obs_key
        Key to compare
    filter_min
        Clump together values with frequency less than filter_min
    """

    ArgAssert(
        (adata.obs[obs_base].dtype == "category") and (adata.obs[obs_key].dtype == "category"),
        "Both obs_key and obs_base have to be categorical"
    )

    old_cats, old_cat_names, old_uq_cats = _get_memberships(adata, obs_base)
    cats, cat_names, uq_cats = _get_memberships(adata, obs_key)

    #cu: CategoricalColorUtil = ColorUtil(adata, obs_key)
    #col_map = cu.get_col_arr()

    plt.rcParams['figure.constrained_layout.use'] = False
    fig = plt.figure(figsize=settings.figsize)
    plw = MplWrap(False, bind_fig=fig, dummy=True)

    _shape = int(np.ceil(cat_names.shape[0] / 4))
    axs = [plt.subplot2grid(shape=(_shape, 4), loc=(int(a / 4), a % 4)) for a in range(cat_names.shape[0])]

    for i, l in enumerate(uq_cats):
        mask = cats == l
        if mask.sum() == 0:
            continue
        freq = [(old_cats[mask] == ol).sum() for ol in old_uq_cats]
        freqd = dict(zip(old_cat_names, freq))
        fmin = int(filter_min * np.array(freq).sum())
        frequency_pie(np.array([]), precomputed_freq=freqd, ax=axs[i], title=cat_names[i], filter_min=fmin)

    plt.axis('off')
    plt.show()
    return None
