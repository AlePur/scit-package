import numpy as np
from typing import Literal
from anndata import AnnData

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Wedge
from matplotlib.figure import Figure
import matplotlib.colors as colors

from .._logging import logging
from ..plotting._helper import MplWrap
from .._utils import ArgAssert

plot_type_type = Literal['mixed', 'single', 'sidebyside', 'circle']


def _pad_zero(A: np.ndarray, offset: bool = False) -> np.ndarray:
    a = A
    c = np.full((a.shape[0] * 2, a.shape[1]), fill_value=np.nan, dtype=a.dtype)
    if offset:
        c[1::2] = a
    else:
        c[0::2] = a
    return c

def heatmap(
        adata: AnnData,
        data: list[np.ndarray],
        plot_type: plot_type_type = 'circle',
        *,
        add_text: bool = True,
        col_threshold: float = 0.3,
        obs_key: str | None = None,
        dendrogram_order: bool = True,
        custom_order: np.ndarray | None = None,
        log_scale: bool = False,
        fontsize: float | None = None,
        vminmax: tuple[float, float] | None = None,
        square_matrix: bool = False,
        cmaps: tuple[str, ...] = ('Blues', 'Reds'),
        show: bool = True
) -> None | Figure:
    """
    Plot heatmap or half-circle plot
    
    Parameters
    ----------
    plot_type
        Type of plot
    layer_indices
        Tuple of layer indices
    dendrogram_order
        Use dendrogram order from `.uns['metabin_community_tree']['order']`
    log_scale
        Scale circle diagram by log transformed counts
    cmaps
        Tuple of Matplotlib colormap names to use
    """
    _ns = "Heatmap of data from more than 2 layers is not supported yet"

    plot_mixed = False
    plot_sidebyside = False
    plot_circle = False
    if plot_type == 'mixed':
        plot_mixed = True
    if plot_type == 'sidebyside':
        plot_sidebyside = True

    if plot_type == 'circle':
        plot_circle = True
    else:
        if log_scale:
            logging.warning("log_scaling does not work for other plot types than 'circle'")

    if custom_order:
        order = np.array(custom_order)
    else:
        if dendrogram_order:
            order = adata.uns['community_tree']['order']
        else:
            order = np.arange(data[0].shape[0])

    data = [d[order] for d in data]
    if square_matrix:
        data = [d[:, order] for d in data]

    if plot_circle:
        mdata = np.ones(data[0].shape)
        #mdatas = [quantized.mdatas[i][:, order] for i in layer_indices]

    if plot_sidebyside:
        data[0] = _pad_zero(data[0])
        data[1] = _pad_zero(data[1], offset=True)

    xlabel = list(order)
    ylabel = list(np.arange(data[0].shape[1]))

    if obs_key is not None:
        xlabel = adata.obs[obs_key].cat.categories[order]  # list(np.arange(matrix.shape[0]))

    if square_matrix:
        ylabel = xlabel.copy()

    plw = MplWrap(show)
    norms = [None, None]
    for inx in [0, 1]:
        if plot_type == 'single':
            if inx == 1:
                break
        if vminmax is None:
            norms[inx] = colors.Normalize(vmin=data[inx].min(), vmax=data[inx].max())
            logging.info(f"For {inx}, min: {data[inx].min()}, max: {data[inx].max()}")
        else:
            norms[inx] = colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])

    if plot_circle:

        s = data[0].shape
        radius = 0.4
        plw.ax.set(xlim=(-radius, radius + s[0] - 1), ylim=(-radius, radius + s[1] - 1))

        if log_scale:
            mdata = [np.log(md) - np.log(md).min() for md in mdata]
            data = [np.log(md) - np.log(md).min() for md in data]

        mdata = [(x.T / (x.max(axis=1))).T for x in mdata]

        #datas = [datas[i]/mdatas[i] for i in range(len(mdatas))]
        #datas = [((x.T) / (x.max(axis=1))).T for x in datas]
        #mdatas = [(md/md.max()) for md in mdatas]

        def getscalarmap(_cmn: str, inx: int) -> cm.ScalarMappable:
            _cm = plt.get_cmap(_cmn)
            return cm.ScalarMappable(norm=norms[inx], cmap=_cm)

        def getradius(i: int, j: int, inx: int) -> float:
            return radius * (mdata[inx][i, j])

        scm0 = getscalarmap(cmaps[0], 0)
        scm1 = getscalarmap(cmaps[1], 1)

        for i in np.arange(s[0]):
            for j in np.arange(s[1]):
                # c = plt.Circle((i, j), radius)
                c = Wedge((i, j), getradius(i, j, 0), 90, 270, fc=scm0.to_rgba(data[0][i, j]))
                c2 = Wedge((i, j), getradius(i, j, 1), 270, 90, fc=scm1.to_rgba(data[1][i, j]))
                plw.ax.add_patch(c)
                plw.ax.add_patch(c2)

        plw.ax.set_aspect('equal')
        plw.remove_axis()
    else:
        _cm = cm.get_cmap(cmaps[0])
        if plot_mixed or plot_sidebyside:
            plw.ax.imshow(data[0].T, _cm, norm=norms[0], origin='lower', aspect='auto', interpolation='nearest')
            _cm = cm.get_cmap(cmaps[1])
            plw.ax.imshow(
                data[1].T,
                _cm,
                norm=norms[1],
                origin='lower',
                aspect='auto',
                interpolation='nearest',
                alpha=0.5 if plot_mixed else 1
            )
        else:
            plw.ax.imshow(data[0].T, _cm, norm=norms[0], origin='lower', aspect='auto', interpolation='nearest')
            if add_text:
                for i in range(data[0].shape[1]):
                    for j in range(data[0].shape[0]):
                        plw.ax.text(
                            i, j, f'{data[0][i, j]:.2f}',
                            ha='center', va='center', fontsize=fontsize,
                            color='white' if data[0][i, j] > col_threshold else 'black'
                        )

    # Show all ticks and label them with the respective list entries
    if plot_sidebyside:
        plw.ax.set_xticks(np.arange(len(xlabel)) * 2, labels=xlabel)
    else:
        plw.ax.set_xticks(np.arange(len(xlabel)), labels=xlabel)
    plw.ax.set_yticks(np.arange(len(ylabel)), labels=ylabel)

    plt.setp(
        plw.ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor"
    )

    plw.fig.tight_layout()
    return plw.show()
