import numpy as np
from anndata import AnnData
from typing import Any

from ..._settings import settings
from ..._utils import ArgAssert
from ...plotting._helper import MplWrap
from ..._logging import logging
from .._griddata import Griddata, _getgrid, _get_griddata_z
import matplotlib.pyplot as plt


RN = 10


def _summarybar(adata: AnnData, ax: Any, scale: int = 1) -> None:
    ArgAssert('summary' in adata.uns.keys(), "Please run toolkit.pl.lsc.make_color_summary first")

    _X = adata.uns['summary'].copy()

    RN = 10
    _X = np.repeat(_X,scale, axis=1)
    ax.imshow(
        np.repeat(_X, RN, axis=0),
        origin='lower'
    )


def _transform_grid(g: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    r = g[0].shape
    return (
        r[0] * (g[0] - g[0].min()) / (g[0].max() - g[0].min()),
        r[1] * (g[1] - g[1].min()) / (g[1].max() - g[1].min())
    )

def _get_axes(summary: bool, gy: int) -> list[Any]:
    axs: list[Any]
    plt.rcParams['figure.constrained_layout.use'] = True
    d= {}
    if summary:
        d['height_ratios'] = [int(gy / RN), 1]
    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw=d,
        figsize=settings.figsize
    )  # type:ignore
    #axs[0].set_aspect(1.0)
    for side in ['bottom', 'right', 'top', 'left']:
        axs[1].spines[side].set_visible(False)
    for ax in axs:
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    return axs


def landscape2d(
        adata: AnnData,
        griddata_key: str,
        feature_name: str,
        *,
        summary: bool = True,
        vminmax: None | tuple[float, float] = None,
        cmap: str = 'gist_earth'
) -> None:
    """
    Plot numerical in 2D landscape embedding
    """

    gd: Griddata = adata.uns['griddata'][griddata_key]
    ArgAssert(not gd["categorical"], "Griddata has to be numerical")
    _z = _get_griddata_z(gd, feature_name)
    grid_x, grid_y = gd["grid"]
    axs = _get_axes(summary, grid_x.shape[1])
    #grid_xt, grid_yt = _transform_grid(griddata.grid)

    if vminmax is not None:
        _vmin, _vmax = vminmax
    else:
        _vmin = None
        _vmax = None

    axs[0].imshow(_z.T, origin='lower', cmap=plt.get_cmap(cmap), vmin=_vmin, vmax=_vmax)

    if summary:
        _summarybar(adata, axs[1], gd["scale_factor"])


def contour2d(
        adata: AnnData,
        griddata_key: str,
        feature_name: str,
        levels: int = 5,
        *,
        summary: bool = True,
        cmap: str = "RdBu_r"
) -> None:
    """
    Plot numerical as contour in landscape embedding
    """

    gd: Griddata = adata.uns['griddata'][griddata_key]
    ArgAssert(gd["categorical"], "Griddata has to be categorical")
    _z = _get_griddata_z(gd, feature_name)
    grid_x, grid_y = gd["grid"]

    axs = _get_axes(summary, grid_x.shape[1])
    grid_xt, grid_yt = _transform_grid(gd["grid"])

    axs[0].contour(grid_xt, grid_yt, _z, levels=levels, linewidths=0.5, colors='k')
    axs[0].contourf(grid_xt, grid_yt, _z, levels=levels, cmap=cmap)

    if summary:
        _summarybar(adata, axs[1], gd["scale_factor"])


def categorical2d(
        adata: AnnData,
        griddata_key: str,
        *,
        summary: bool = True
) -> None:
    """
    Plot categorical in landscape embedding
    """
    gd: Griddata = adata.uns['griddata'][griddata_key]
    ArgAssert(gd["categorical"], "Griddata has to be categorical")

    grid_x, grid_y = gd["grid"]
    _z = gd["z"]

    axs = _get_axes(summary, grid_x.shape[1])
    rav_z = _z.ravel()

    colors = gd["color_map"][rav_z]
    colors[rav_z == -1] = np.array([0,0,0])

    axs[0].imshow(
        np.transpose(
            colors.reshape((grid_x.shape[0], grid_y.shape[1], 3)),
            axes=(1, 0, 2)
        ),
        origin='lower'
    )
    if summary:
        _summarybar(adata, axs[1], gd["scale_factor"])
