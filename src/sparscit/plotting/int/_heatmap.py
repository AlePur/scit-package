from anndata import AnnData
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from .._helper import MplWrap
from ..._utils import _get_memberships, ArgAssert
from ..._logging import logging

def heatmap(
        data: np.ndarray,
        labels: list[str],
        *,
        ylabels: list[str],
        vminmax: tuple[float, float] | None = None,
        square_matrix: bool = True,
        show_ylabels: bool = True,
        show_xlabels: bool = True,
        show_entries: bool = False,
        naked: bool = False,
        cmap: str = 'Blues',
        show: bool = True
) -> None | Figure:
    """
    Create heatmap.

    Parameters
    ----------
    data
    labels
    vminmax
        Min, max tuple for normalization of colors
    square_matrix
        Whether to plot a square matrix (axis lengths are equal)
    show_ylabels
        show y-labels
    cmap
        Colormap name to use
    show

    """
    xlabel = labels
    if ylabels != None:
        ylabel = ylabels
    else:
        ylabel = list(np.arange(data.shape[1]))
        if square_matrix:
            ylabel = labels

    plw = MplWrap(show)
    norms = None
    if vminmax is None:
        norms = colors.Normalize(vmin=data.min(), vmax=data.max())
        logging.info(f"min: {data.min()}, max: {data.max()}")
    else:
        norms = colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])

    _cm = cm.get_cmap(cmap)
    plw.ax.imshow(data.T, _cm, norm=norms, origin='lower', aspect='auto', interpolation='nearest')

    if show_entries:
        # Loop over data dimensions and create text annotations.
        # Note: We must transpose data indices because imshow is plotted as data.T
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                
                # Determine text color based on background intensity for readability
                # (Simple heuristic: white text on dark cells, black on light)
                text_color = "white" if norms(val) > 0.5 else "black"
                
                plw.ax.text(i, j, f"{val:.2f}",
                            ha="center", va="center", color=text_color)


    # Show all ticks and label them with the respective list entries
    if show_xlabels:
        plw.ax.set_xticks(np.arange(len(xlabel)), labels=xlabel)
        plt.setp(
            plw.ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor"
        )
    else:
        plw.ax.set_xticks([],[])
    if show_ylabels:
        plw.ax.set_yticks(np.arange(len(ylabel)), labels=ylabel)
    else:
        plw.ax.set_yticks([],[])

    if naked:
        plw.ax.set_axis_off()
    
    return plw.show()

