from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from anndata import AnnData

from ..._utils import ArgAssert
from .._griddata import _getgrid, Griddata, _get_griddata_z


def landscape3d(
        griddata: Griddata,
        feature_name: str,
        vminmax: tuple[float, float] | None = None,
        *,
        shade_data: Griddata | None = None,
        shade_vminmax: tuple[float, float] | None = None,
        shade_qmax: float | None = None,
        cmap: str = 'gist_earth',
        shade_cmap: str = 'Reds',
        antialiased: bool = False,
        extra_shade_weight: float = 1.0,
        azaltdeg: tuple[int, int] = (270, 45),
        ticksize: float | None = None,
        negative_yaxis: bool = True,
        show_xaxis: bool = False,
        fontname: str = 'Helvetica',
        figsize: tuple = (6, 6),
        x_aspect: int = 1,
        label_pad: float = 2,
        show: bool = True
) -> None | Figure:
    """
    Plot numerical as 3d in landscape embedding

    Parameters
    ----------
    griddata
        Griddata for height
    vminmax
        Tuple of vmin and vmax for color normalization
    shade_vminmax
        vminmax for shade_data
    shade_data
        Optional :class:`Griddata` for coloring the 3d surface
    """
    gd = griddata
    ArgAssert(not gd["categorical"], "Griddata has to be numerical")
    grid_x, grid_y = gd["grid"]
    _z = _get_griddata_z(gd, feature_name)

    ax: Axes3D
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=figsize)
    ax.set_box_aspect((x_aspect, 1, 1))

    ls = LightSource(azaltdeg[0], azaltdeg[1])
    if vminmax is None:
        vmin, vmax = (_z.min(), _z.max())
    else:
        vmin, vmax = vminmax

    rgb = ls.shade(_z, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax, vert_exag=0.1, blend_mode='soft')

    if shade_data is not None:
        shade_z = _get_griddata_z(shade_data, feature_name)
        ArgAssert(
            gd["grid"][0].shape == shade_data["grid"][0].shape,
            'Shade griddata and height griddata must have same size'
        )
        if shade_vminmax is None:
            svmin, svmax = (shade_z.min(), shade_z.max())
        else:
            svmin, svmax = shade_vminmax
        if shade_qmax is not None:
            svmax = np.quantile(shade_z.ravel(), shade_qmax)
        print(f"RNA range: ({svmin}, {svmax})")
        extra_shade = LightSource(azaltdeg[0], azaltdeg[1]).shade(
            shade_z, cmap=plt.get_cmap(shade_cmap), vmin=svmin, vmax=svmax, vert_exag=0.1, blend_mode='soft'
        )
        rgb = (rgb + extra_shade*extra_shade_weight) / (1+extra_shade_weight)
    surf = ax.plot_surface(
        grid_x,
        grid_y,
        _z,
        rstride=1,
        cstride=1,
        facecolors=rgb,
        linewidth=0,
        antialiased=antialiased,
        shade=False,
        vmin=vmin,
        vmax=vmax
    )
    ax.set_zlim(vmin, vmax)
    if ticksize is not None:
        ax.tick_params(axis='both', which='major', pad=label_pad, labelsize=ticksize, labelfontfamily=fontname)
        ax.tick_params(axis='both', which='minor', pad=label_pad, labelsize=ticksize, labelfontfamily=fontname)
    if negative_yaxis:
        lbl = ax.get_yticklabels()
        ax.set_yticklabels([l.get_text()[1:] for l in lbl])
    if not show_xaxis:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax.view_init(elev=azaltdeg[1], azim=azaltdeg[0])
    if show:
        plt.show()
    else:
        return plt.gcf()