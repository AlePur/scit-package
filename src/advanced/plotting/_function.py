import numpy as np
from anndata import AnnData
from matplotlib.figure import Figure
import scipy.sparse as sps

from ..._logging import logging
from ..._utils import ArgAssert
from ...plotting._helper import MplWrap
from .._griddata import Griddata, _getgrid
from ...tools._layers import LayerConfig
from ._2d import _summarybar, _transform_grid

from itertools import groupby


def slice(
        griddatas: list[Griddata],
        slice_coord: tuple[complex, complex] | tuple[int, int],
        griddata_categorical: Griddata | None = None,
        *,
        include_std: bool = False,
        show: bool = True,
        labels: list[str] | None = None
) -> None | Figure:
    # TODO: Broken. old griddatas
    """
    Plot slices.

    Parameters
    ----------
    griddatas
    slice_coord
    griddata_categorical
    include_std
    show
    labels

    Returns
    -------

    """
    is_complex = False
    if isinstance(slice_coord[0], complex):
        slice_coord = (int(slice_coord[0].imag), int(slice_coord[1].imag))
        is_complex = True
    else:
        slice_coord = (int(slice_coord[0]), int(slice_coord[1]))

    ArgAssert(
        np.array([gd.grid == griddatas[0].grid for gd in griddatas]).all(),
        'All grids must be same size'
    )

    if labels is not None:
        ArgAssert(len(labels) == len(griddatas), 'Length of labels has to be equal to length of griddatas')

    if griddata_categorical is not None:
        ArgAssert(
            tuple(np.array(griddata_categorical.grid[0].shape) * griddatas[0].scale_factor)
            == griddatas[0].grid[0].shape,
            'Griddatas and griddata_categorical must have same shape'
        )
        ArgAssert(griddata_categorical.categorical, "Griddata has to be categorical")
        _pl = MplWrap(True)
        _pl.ax.imshow(griddata_categorical.z.T[::-1])
        sx = [slice_coord[0], slice_coord[1], slice_coord[1], slice_coord[0], slice_coord[0]]
        _pl.ax.plot(
            [0, 0, griddata_categorical.z.shape[0] - 1, griddata_categorical.z.shape[0] - 1, 0] if is_complex else sx,
            [0, 0, griddata_categorical.z.shape[1] - 1, griddata_categorical.z.shape[1] - 1, 0] if not is_complex else sx,
            c='r'
        )

        _pl.show()

    plw = MplWrap(show=show)
    slice_coord *= griddatas[0].scale_factor
    previous_data = None
    for i, gd in enumerate(griddatas):
        kwarg = {}
        if labels is not None:
            kwarg['label'] = labels[i]
        if not is_complex:
            data = gd.z[slice_coord[0]:slice_coord[1]].sum(axis=0)[::-1]
        else:
            data = gd.z[:, slice_coord[0]:slice_coord[1]].sum(axis=1)
        if include_std and i % 2 == 1:
            # Fill the s
            plw.ax.errorbar(np.arange(data.shape[0]), previous_data, yerr=data, capsize=3)
        else:
            previous_data = data
            plw.ax.plot(data, **kwarg)

    if is_complex:
        plw.ax.set_xlabel('Landscape')
    else:
        plw.ax.set_xlabel('Time')
    plw.ax.set_ylabel('Signal')
    if labels is not None:
        plw.fig.legend()
    plw.despine()
    return plw.show()