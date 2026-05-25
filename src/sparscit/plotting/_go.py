from typing import Literal
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import numpy as np
import tempfile
import pandas as pd
import polars as pl

from ..tools._goterm import GOEA
from ._helper import MplWrap
from ._bar import barplot, smart_barplot

_plot_type = Literal['simple', 'smart']
_goea_mode = Literal['enriched_only', 'purged_only', 'both']


def _goea_filter(results: GOEA, mode: _goea_mode):
    fr = []
    for r in results.res:
        if mode == 'enriched_only':
            if r.enrichment == 'p':
                continue
        elif mode == 'purged_only':
            if r.enrichment == 'e':
                continue

        fr.append(r)
    return fr


def goea(
        goea_results: GOEA | pl.DataFrame,
        n_top: int | None = 5,
        *,
        color: str = "tab:blue",
        mode: _goea_mode = 'both',
        plot_type: _plot_type = 'simple',
        title: str = 'GO',
        text_size: float = 10,
        wrap_width: int = 50,
        show: bool = True,
) -> None | list:
    """
    Plot GO Enrichment Analysis results.

    Parameters
    ----------
    goea_results
        :class:`GOEA`
    plot_type
        Either 'smart' or 'simple'
    n_top
        Top results to plot (sorted based on pval)
    mode
        Plot filtered results, one of 'enriched_only', 'purged_only' or 'both'
    title
        Title of plot

    """
    _df = goea_results.df if isinstance(goea_results, GOEA) else goea_results
    _df = pl.DataFrame(_df)
    fs = []
    for n in _df['NS'].unique():

        sorted = _df.filter(pl.col('NS')==n).sort('p_corrected')
        if mode == 'both':
            pass
        elif mode == 'enriched_only':
            sorted = sorted.filter(pl.col('enriched'))
        elif mode == 'purged_only':
            sorted = sorted.filter(pl.col('enriched').not_())
        if n_top is not None:
            taken = sorted.top_k(n_top, by='p_corrected',reverse=True)['name']
            sorted = sorted.filter(pl.col('name').is_in(taken))
        pvals = -np.log10(sorted['p_corrected'].to_numpy())

        if plot_type == 'simple':
            fs.append(
                barplot(
                    pvals,
                    sorted['name'],
                    sorted['enriched'],
                    (color, 'tab:red'),
                    title=n,
                    h=True,
                    ten_power=True,
                    text_size=text_size,
                    show=show
                )
            )
        else:
            fs.append(
                smart_barplot(
                    pvals,
                    sorted['name'],
                    sorted['enriched'],
                    (color, 'tab:red'),
                    title=n,
                    ten_power=True,
                    text_size=text_size,
                    wrap_width=wrap_width,
                    show=show
                )
            )

    if not show:
        return fs

        


def goea_tree(
        goea_results: GOEA,
        mode: _goea_mode = 'enriched_only'
) -> None:
    """
    Plot GO Enrichment Analysis results in tree-format.

    Parameters
    ----------
    goea_results
        :class:`GOEA`
    mode
        Plot 'enrichment_only', 'purged_only' or 'both'
    """
    from goatools.godag_plot import plot_gos, plot_results, plot_goid2goobj

    res = _goea_filter(goea_results, mode)

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_results(os.path.join(tmpdir, "{NS}.png"), res)

        f = []
        for (dirpath, dirnames, filenames) in os.walk(tmpdir):
            f.extend(filenames)
            break

        for file in f:
            img = plt.imread(os.path.join(tmpdir, file))
            plw = MplWrap(True)
            plw.ax.imshow(img)
            plw.ax.set_title(file)
            plw.ax.axis('off')
            plt.show()