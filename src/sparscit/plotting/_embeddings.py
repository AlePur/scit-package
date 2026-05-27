from anndata import AnnData
from matplotlib.pyplot import Figure, cm, Axes
import numpy as np
from matplotlib.colors import Colormap
import matplotlib.patheffects as pe

from ._helper import MplWrap, ColorUtil, CategoricalColorUtil, NumColorUtil
from .._utils import ArgAssert
from .._logging import logging


def _draw_distance(
        ax: Axes, xy1: np.ndarray, xy2: np.ndarray, dist: np.ndarray,
        *,
        line_width: float,
        text_size: float
) -> None:
    """
    Draw an arrow with a distance label between two points on an axes.

    Parameters
    ----------
    ax
        Matplotlib axes to draw on
    xy1
        Start point coordinates
    xy2
        End point coordinates
    dist
        Distance value to display as label
    line_width
        Width of the arrow line
    text_size
        Font size of the distance label
    """
    ax.annotate(
        "",
        xy=xy1,
        xycoords='data',
        xytext=xy2,
        textcoords='data',
        arrowprops=dict(
            arrowstyle='->', color='black', alpha=1,
            linewidth=line_width
        )
    )

    ax.text(
        (xy1[0] + xy2[0]) / 2,
        (xy1[1] + xy2[1]) / 2,
        '{0:.2g}'.format(dist),
        ha='center',
        va='center',
        color="black",
        path_effects=[pe.withStroke(linewidth=int(np.clip(text_size / 5, 1, 5)), foreground="white")],
        fontsize=text_size
    )

def embedding2d(
        adata: AnnData,
        embedding_key: str,
        color: str | None = None,
        *,
        first_pcs_only: bool = False,
        size: float = 3,
        alpha: float = 1,
        sort_order: bool = True,
        vminmax: tuple[float, float] | None = None,
        cmap: Colormap | str | None = None,
        black_background: bool = False,
        force_legend: bool = False,
        coax_numerical: bool = False,
        title: str | None = None,
        draw_ontop: Figure | None = None,
        mask: np.ndarray | None = None,
        mask_style_dict: dict = {
            'color': 'gray',
            'size': 1,
            'alpha': 0.5
        },
        include_arrows: bool = False,
        arrow_style_dict: dict = {
            'line_width': 8,
            'text_size': 10
        },
        show: bool = True,
) -> Figure | None:
    """\
    Plot a 2D embedding

    Parameters
    ----------
    adata
        AnnData object with embedding in `.obsm`
    embedding_key
        Key at which embedding is stored in `adata.obsm`
    color
        Column in `adata.obs` used for coloring
    first_pcs_only
        If True, the function can plot embeddings with more than 2 dimensions.
        It simply plots the first two dimensions
    size
        Size of one cell on the plot
    alpha
        Transparency of one cell on the plot
    sort_order
        If color is numerical, sorting will place higher values on top of lower ones
    vminmax
        Tuple of (vmin, vmax) for numerical plotting
    cmap
        Matplotlib colormap for plotting
    black_background
        Use dark background
    force_legend
        Always show legend, even if too large
    coax_numerical
        Make categorical into numerical if possible
    title
        Title of plot
    draw_ontop
        Draw on top of provided figure
    mask
        Mask cells based on boolean array
    mask_style_dict
        Cells that are not selected by mask are given these style parameters
    include_arrows
        Include arrows from `.uns['arrows']`
    arrow_style_dict
        Arrow styles kwargs
    show
        Show plot

    Returns
    -------
    :class:`matplotlib.figure.Figure` or None
        Figure object if `show=False`, otherwise None
    """
    arrow_style_dict = arrow_style_dict.copy()
    mask_style_dict = mask_style_dict.copy()
    ArgAssert(embedding_key in adata.obsm.keys(), "embedding_key not in .obsm")
    if color is not None:
        ArgAssert(color in adata.obs.keys(), "color not in .obs")
    if not first_pcs_only:
        ArgAssert(
            adata.obsm[embedding_key].shape[1] == 2,
            "the embedding is not 2 dimensional. If you still want to plot it, set first_pcs_only to True"
        )
        X = adata.obsm[embedding_key]
    else:
        X = adata.obsm[embedding_key][:, :2]
    assert X.shape[1] == 2

    plw = MplWrap(show=show, bind_fig=draw_ontop)

    if coax_numerical:
        col_util = ColorUtil(adata, '', adata.obs[color])
    else:
        col_util = ColorUtil(adata, color)
    col_arr = col_util.get_col_arr()

    if isinstance(col_util, CategoricalColorUtil):
        if mask is not None:
            plw.ax.scatter(
                X[:, 0],
                X[:, 1],
                c=mask_style_dict['color'],
                s=mask_style_dict['size'],
                alpha=mask_style_dict['alpha']
            )
            plw.ax.scatter(X[mask][:, 0], X[mask][:, 1], c=np.array(col_arr)[mask], s=size, alpha=alpha)
        else:
            plw.ax.scatter(X[:, 0], X[:, 1], c=col_arr, s=size, alpha=alpha)
        h = col_util.get_legend()

        if (len(h) < 40) or force_legend:
            plw.fig.legend(handles=h, bbox_to_anchor=(1.0, 0.5), loc="center left", borderaxespad=0)
        else:
            logging.warning(
                "Skipping legend due to too many (>40) categoricals. Set force_legend = True if you still "
                "want the legend"
            )
    elif isinstance(col_util, NumColorUtil):

        if mask is not None:
            raise NotImplementedError("Mask cannot be used with numerical values")

        if cmap is None:
            _cm = cm.get_cmap('viridis')
        else:
            _cm = cmap  # type:ignore
        if vminmax is None:
            vmin, vmax = (col_arr.min(), col_arr.max())  # type:ignore
        else:
            vmin, vmax = vminmax  # type:ignore

        if (sort_order):
            o = np.array(adata.obs[color].argsort())
        else:
            o = np.arange(adata.shape[0])
        sc = plw.ax.scatter(X[o][:, 0], X[o][:, 1], c=col_arr[o], s=size, alpha=alpha, vmin=vmin, vmax=vmax, cmap=_cm)
        cbar = plw.fig.colorbar(sc, ax=plw.ax)
        cbar.solids.set(alpha=1)  # type:ignore
    else:
        if mask is not None:
            raise NotImplementedError("Mask cannot be used here")
        plw.ax.scatter(X[:, 0], X[:, 1], c=col_arr, s=size, alpha=alpha)

    if black_background:
        plw.ax.set_facecolor("black")

    if title is not None:
        plw.ax.set_title(
            label=title,
            fontweight='bold'
        )

    if include_arrows:
        for d in adata.uns['arrows']:
            _draw_distance(plw.ax, *d, **arrow_style_dict)

    plw.arrowed_spines()
    return plw.show()
