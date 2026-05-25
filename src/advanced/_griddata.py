import scipy as sp
import numpy as np
from typing import Literal, Any, Self
import pandas as pd
from pandas.api.types import is_numeric_dtype
from ._gaussian import gaussian_filter
from jax.scipy.interpolate import RegularGridInterpolator as jRegularGridInterpolator
import jax.numpy as jnp
from functools import partial
from anndata import AnnData
from typing import cast
from jax import jit, vmap

from ._median import median_filter
from ..tools._layers import LayerConfig
from .._utils import ArgAssert
from ..plotting._helper import MplWrap, ColorUtil, CategoricalColorUtil
from .._logging import logging
from ._grid import _get_image_data, _min_filter, _add_to_gd_uns
from typing import TypedDict


class Griddata(TypedDict):
    z: np.ndarray # 3D: [feature, x, y]
    metadata: np.ndarray # 2D
    categorical: bool
    grid: list[np.ndarray]
    min_cells: int
    feature_names: np.ndarray | None
    scale_factor: int | None # 1
    color_map: dict | None


_method_type = Literal['linear', 'nearest', 'cubic']


def _getgrid(adata: AnnData, jax: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[jnp.ndarray, jnp.ndarray]:
    ArgAssert('grid' in adata.uns.keys(), "Please create a grid first")
    grd = adata.uns['grid']['xy']
    if jax:
        grd = (jnp.array(grd[0]), jnp.array(grd[1]))
    return grd


def _get_lsc(adata: AnnData) -> np.ndarray:
    ArgAssert('X_landscape' in adata.obsm.keys(), "No landscape found")
    return adata.obsm['X_landscape']


def make_grid(
        adata: AnnData,
        resolution: tuple[int, int] = (300, 300)
) -> None:
    """
    Makes a grid based on `.obsm['X_landscape']`

    Parameters
    ----------
    resolution
        Tuple of integers specifying the resolution of grid

    Returns
    -------
    `adata.uns['grid']`
    """
    X = _get_lsc(adata)

    grid = np.mgrid[X[:, 0].min():X[:, 0].max():complex(0, resolution[0]),
           X[:, 1].min():X[:, 1].max():complex(0, resolution[1])]  # type:ignore
    adata.uns['grid'] = {
        'resolution': list(resolution),
        'xy': grid
    }

def _get_griddata_z(
      gd: Griddata,
      feature_name: str
) -> np.ndarray:
    _nf = "It seems that there is no feature with the index as provided in feature_name"
    
    assert gd["feature_names"] is not None, "feature_names missing"

    try:
        m = gd["feature_names"] == feature_name
    except Exception as e:
        logging.warning(_nf)
        raise e

    if m.sum() == 0:
        raise ValueError(_nf)

    return gd["z"][np.argmax(m)]

def griddata_numerical(
        adata: AnnData,
        layer: LayerConfig,
        min_cells: int,
        *,
        use_cached_mask: bool = True
):
    """
    Make :class:`Griddata` from numerical feature of adata.

    Parameters
    ----------
    obs_key
        Key for numerical in .obs
    gauss_kernel
        If this is a tuple of integers, use gaussian blur for smoothing.
        This is the sigma (standard deviation) parameter of the gaussian kernel
    nan_fill
        Replace nan with this value
    min_cells
        Min cells to create a pixel
        
    Returns
    -------
    :class:`Griddata`
    """

    fns = layer.get_feature_names()
    X = _get_lsc(adata)
    X = jnp.array(X)
    z = layer.transform(adata, use_cached_mask=use_cached_mask)
    z = jnp.array(z.toarray())

    grid_x, grid_y = _getgrid(adata, jax=True)
    Z = _get_image_data(X, z, (grid_x, grid_y))
    Z_m = _get_image_data(X, z, (grid_x, grid_y), metadata=True)
    del z
    Z = _min_filter(Z, Z_m, min_cells)
    Z_m = _min_filter(Z_m, Z_m, min_cells)
    
    gd: Griddata = dict(
        z=np.array(Z),
        metadata=np.array(Z_m),
        categorical=False,
        grid=[np.array(grid_x), np.array(grid_y)],
        feature_names=fns,
        min_cells=min_cells,
        scale_factor=1,
        color_map=None
    )
    
    k = layer.layer
    _add_to_gd_uns(adata, k, gd)
    logging.info(f"Added Griddata to .uns[{k}]")

    #.min_filter(min_cells).median_filter(msize).gaussian_filter().nan_fill(nan_fill)

    # MULTI
    # coeff[0]*gd_1.z + coeff[1]*gd_2.z



def griddata_categorical(
        adata: AnnData,
        obs_key: str,
        min_cells: int
):
    """
    Make :class:`Griddata` from numerical feature of adata.

    Parameters
    ----------
    adata
    obs_key

    Returns
    -------
    :class:`Griddata`
    """

    ArgAssert(obs_key in adata.obs.keys(), "Key not found in obs")

    cu: CategoricalColorUtil = ColorUtil(adata, obs_key)  # type:ignore
    ArgAssert(isinstance(cu, CategoricalColorUtil), "key is not categorical")

    col_map = cu.get_categorical_rgb()
    z = adata.obs[obs_key].cat.codes.values

    X = _get_lsc(adata)
    X = jnp.array(X)
    z = jnp.array(z)

    grid_x, grid_y = _getgrid(adata, jax=True)
    Z = _get_image_data(X, z, (grid_x, grid_y), categorical=True)
    Z_m = _get_image_data(X, z, (grid_x, grid_y), metadata=True)
    Z = _min_filter(Z, Z_m, min_cells, fill_value=-1)
    Z_m = _min_filter(Z_m, Z_m, min_cells)
    
    gd: Griddata = dict(
        z=np.array(Z),
        metadata=np.array(Z_m),
        categorical=True,
        grid=[np.array(grid_x), np.array(grid_y)],
        feature_names=None,
        min_cells=min_cells,
        scale_factor=1,
        color_map=col_map
    )
    
    k = f"cat_{obs_key}"
    _add_to_gd_uns(adata, k, gd)
    logging.info(f"Added Griddata to .uns[{k}]")


#def lazy_shuffle(self):
#    shuffler = np.random.RandomState(seed=self.shuffle_seed)
#    self._ix = shuffler.randint(0, self.X.shape[0], (self.X.shape[0],))
#
#def get_slice(self, k: str, keep_dims: bool = False, use_gauss=False):
#    _X = self.X
#    if use_gauss:
#        _X = self.X_gauss
#    _X = _X[:, :, :, np.array(self.k_dict[k])]
#    if keep_dims:
#        return _X[np.newaxis, :, :, :]
#    return _X
#
#def return_combinations(self, i_comparison_target: int, use_gauss=False) -> list[tuple[np.ndarray, np.ndarray]]:
#    from itertools import product
#    _X = self.X
#    if use_gauss:
#        _X = self.X_gauss
#    x = np.arange(_X.shape[3])
#    combs = list(iter(product((x[x != i_comparison_target]), (x[i_comparison_target][np.newaxis]))))
#    return [
#        (
#            _X[:, :, :, comb[0]],
#            _X[:, :, :, comb[1]]
#        )
#        for comb in combs
#    ]


def griddata_apply_transformations(
        adata,
        griddata_key: str,
        gauss_kernel: tuple[float, float] | None = None,
        median_filter_size: tuple[int, int] | None = (3, 3),
        n_median_passes: int = 2,
        upscale_factor: float = 1,
        data_scalar: float = 1.0,
        *,
        max_up: float = 20.0,
        all_feature_q_norm: float | None = None,
        data_range_q: float = 1.0,
        winsor_qs_low: float | None = None,
        winsor_qs_high: float | None = None,
        winsor_q_is_p_max: bool = False,
        winsor_set_low_zero: bool = False,
        nan_fill: float = 0.0
) -> None:
    gd: Griddata = adata.uns['griddata'][griddata_key]

    ArgAssert(not gd['categorical'], "Categorical griddata is not supported")
    ArgAssert(np.issubdtype(type(upscale_factor), np.integer), 'Scale factor must be integer')
    missing = gd['metadata'] == 0
    gd_grid = gd["grid"]
    gd_scale_factor = gd["scale_factor"]

    extra_kwarg = {}

    # Upscale
    if upscale_factor != 1.0:

        xs = gd_grid[0][:, 0]
        ys = gd_grid[1][0]
        new_res = (
            int(upscale_factor * xs.shape[0]),
            int(upscale_factor * ys.shape[0])
        )
        eps = float("-1e-8")
        grid = np.mgrid[xs.min()-eps:xs.max()+eps:complex(0, new_res[0]), ys.min()-eps:ys.max()+eps:complex(0, new_res[1])]
        
        logging.info(f'Scaling by {upscale_factor}x: from {gd_grid[0].shape} to {grid[0].shape}')
        
        gd_grid = grid
        gd_scale_factor = upscale_factor
        extra_kwarg = extra_kwarg | dict(
            upscale_old_xs=jnp.array(xs),
            upscale_old_ys=jnp.array(ys),
            upscale_new_grid=jnp.array(grid)
        )

    _X = _apply_transformations(
        jnp.array(gd["z"], copy=True),
        jnp.array(missing),
        gauss_kernel=gauss_kernel,
        median_filter_size=median_filter_size,
        n_median_passes=n_median_passes,
        all_feature_q_norm=all_feature_q_norm,
        winsor_qs_low=winsor_qs_low,
        max_up=max_up,
        data_range_q=data_range_q,
        winsor_qs_high=winsor_qs_high,
        winsor_q_is_p_max=winsor_q_is_p_max,
        winsor_set_low_zero=winsor_set_low_zero,
        gauss_nan_fill=nan_fill,
        **extra_kwarg
    )

    logging.info(f'Added Griddata to .uns[t_{griddata_key}]')
    adata.uns['griddata'][f"t_{griddata_key}"] = gd | {'z': np.array(_X * data_scalar), 'scale_factor': gd_scale_factor, 'grid': gd_grid}
    

@partial(jit, static_argnames=[
    'gauss_kernel',
    'median_filter_size',
    'n_median_passes',
    'all_feature_q_norm',
    'winsor_qs_low',
    'winsor_qs_high',
    'winsor_q_is_p_max',
    'winsor_set_low_zero',
    'gauss_nan_fill'
    ])
def _apply_transformations(
        _X: jnp.ndarray,
        missing: jnp.ndarray,
        *,
        upscale_old_xs: jnp.ndarray | None = None,
        upscale_old_ys: jnp.ndarray | None = None,
        upscale_new_grid: jnp.ndarray | None = None,
        gauss_kernel: tuple[float, float] | None = None,
        median_filter_size: tuple[int, int] | None = (3, 3),
        n_median_passes: int = 2,
        #set_data_range: tuple[float, float] | None = None,
        data_range_q: float = 1.0,
        all_feature_q_norm: float | None = None,
        max_up: float = 20.0,
        winsor_qs_low: float | None = None,
        winsor_qs_high: float | None = None,
        winsor_q_is_p_max: bool = False,
        winsor_set_low_zero: bool = False,
        gauss_nan_fill: float = 0.0
) -> jnp.ndarray:
    _X = jnp.where(missing[None, :], np.nan, _X) # Fill nan later???

    # Median filter
    if median_filter_size is not None:
        for _ in range(n_median_passes):
            _X = jnp.where(missing[None, :], np.nan, _X)
            _X = median_filter(_X.astype(np.float32)[:,:,:,None], median_filter_size).astype(np.float32)[:,:,:,0]
        _X = jnp.where(missing[None, :], 0, _X)

    # Gauss
    if gauss_kernel is not None:
        l = dict(
            mode='edge', axes=np.array([1, 2])
        )
        _X = jnp.where(missing[None, :], 0, _X)
        _X = gaussian_filter(_X, gauss_kernel, **l)  # z is gaussed, missing values are zero
        # now we need to scale up the small values => imputation
        norm = gaussian_filter((~missing).astype(np.float32), gauss_kernel, **(l | {'axes': np.array([0,1])}))

        nanmask = jnp.isclose(np.array(0.0, dtype=np.float32), norm, atol=1e-05)
        norm = jnp.where(nanmask, 1.0, norm) #        norm[nanmask] = 1.0
        _X = _X / norm[np.newaxis, :, :]
        _X = jnp.where(nanmask[None, :], gauss_nan_fill, _X)

    if all_feature_q_norm is not None:
        # winsor
        _max = jnp.quantile(_X, data_range_q)
        _X = jnp.where(_X > _max, _max, _X)
        #minmax
        _X = ((_X - _X.min()) / (_max - _X.min())) #* (set_data_range[1] - set_data_range[0]) + set_data_range[0]

        scaler = 1.0 / jnp.quantile(_X.reshape(_X.shape[0], -1), all_feature_q_norm, axis=1)
        scaler = jnp.where(scaler > max_up, max_up, scaler)
        _X = (scaler * _X.T).T

    # Winsorization
    # A type of winsorization is already done with set_data_range, 
    # But this one is beneficial since all_feature_q_norm comes before it
    # And thus, all features will be affected in a similar way
    if winsor_qs_low is not None or winsor_qs_high is not None:

        if winsor_qs_high is not None:
            if winsor_q_is_p_max:
                q0 = winsor_qs_high * _X.max()
            else:
                q0 = jnp.quantile(_X, winsor_qs_high)

            _X = jnp.where(_X > q0, q0, _X)
        if winsor_qs_low is not None:
            if winsor_q_is_p_max:
                q0 = winsor_qs_low * _X.max()
            else:
                q0 = jnp.quantile(_X, winsor_qs_low)

            _X = jnp.where(_X < q0, 0 if winsor_set_low_zero else q0, _X)

    _X = ((_X - _X.min()) / (_X.max() - _X.min()))
    
    # Upscale
    if upscale_new_grid is not None:

        def interp_grid(i: jnp.ndarray):
            interp = jRegularGridInterpolator(
                (
                    upscale_old_xs,
                    upscale_old_ys
                ),
                _X[i]
            )
            return interp((upscale_new_grid[0], upscale_new_grid[1]))
        _X = vmap(interp_grid)(jnp.arange(_X.shape[0]))

    return _X.astype(np.float32)