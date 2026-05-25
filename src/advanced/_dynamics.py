from typing import Callable
import numpy as np
import scipy.stats as sts
from anndata import AnnData
import polars as pl
from jax import vmap
import jax.numpy as jnp
from jax._src.numpy import ufuncs

from ..tools._layers import LayerConfig
from ._griddata import Griddata
from .._utils import ArgAssert
from .._logging import PBar

def _cor(_a, _times, weights: np.ndarray | None):
        # return jnp.corrcoef(_a, _times, rowvar=False)
        # Copied from jnp.corrcoef
        c = jnp.cov(jnp.c_[_a, _times].T, aweights=weights)
        d = jnp.diag(c)
        stddev = ufuncs.sqrt(ufuncs.real(d)).astype(c.dtype)
        c = c / stddev[:, None] / stddev[None, :]
        return c[1,0]

def _cov_std(_a, _times, weights: np.ndarray | None):
        # return jnp.corrcoef(_a, _times, rowvar=False)
        # Copied from jnp.corrcoef
        c = jnp.cov(jnp.c_[_a, _times].T, aweights=weights)
        d = jnp.diag(c)
        stddev = ufuncs.sqrt(ufuncs.real(d)).astype(c.dtype)
        # c = c / stddev[:, None] / stddev[None, :] -> c[1,0]
        return c[1,0], stddev[0], stddev[1]

jax_pearson_one = vmap(_cor, (0, None, None))
jax_pearson_two = vmap(_cor, (0, 0, None))
jax_pearson_two_wweights = vmap(_cor, (0, 0, 0))
jax_pearson_two_wweights_raw = vmap(_cov_std, (0, 0, 0))

def np_correlation_wrap(
        X: np.ndarray, Y: np.ndarray, 
        spearman: bool = False,
        *,
        weights: np.ndarray | None = None,
        return_raw: bool = False
):
    assert len(X.shape) == 2
    assert len(Y.shape) <= 2
    if spearman:
        X = sts.rankdata(X, axis=1)
    X = jnp.array(X)
    Y = jnp.array(Y)
    if len(X.shape) > len(Y.shape):
        if return_raw:
            raise NotImplementedError()
        r = jax_pearson_one(X, Y, None)
    else:
        if weights is not None:
            weights = jnp.array(weights)
            if return_raw:
                r, s1, s2 = jax_pearson_two_wweights_raw(X, Y, weights)
                return np.array(r), np.array(s1), np.array(s2)
            else:
                r = jax_pearson_two_wweights(X, Y, weights)
        else:
            if return_raw:
                raise NotImplementedError()
            r = jax_pearson_two(X, Y, None)
    return np.array(r)

def dynamics_add_expression_data(
        dyn_df: pl.DataFrame,
        feature_names: np.ndarray | None = None,
        activity_matrix: np.ndarray | None = None,
) -> None:
    ix = np.argsort(pl.DataFrame)

    def geti(n):
        f = np.searchsorted(feature_names[ix], n)
        return ix[f]

    dyn_df = dyn_df.with_columns(
        (
            pl.struct(["group", "var_name"]).map_batches(
                lambda x: activity_matrix[geti(x.struct.field("var_name")), x.struct.field("group")]
            )
        ).alias("exp")
    )
    return dyn_df


def _correlation_internal(
        z0, z1, 
        features,
        spearman: bool,
        ignore_zeros: bool , signal_weighting: bool, return_raw: bool
):
    def t(x):
        _x = np.abs(x)
        assert _x.max() <= 1 and _x.min() >= 0
        return _x ** 2
        #return ((1.0 / x.max(axis=1)) * x.T).T ** 2
    res = None
    
    # if ignore_zeros or signal_weighting:
    if ignore_zeros:
        ignore_mask = ((z0 != 0) * (z1 != 0)).astype(np.int32)
        res = np_correlation_wrap(z0, z1, spearman=spearman, weights=ignore_mask)
    elif signal_weighting:
        
        weights_matrix = (t(z0) + t(z1)).astype(np.float32)
        # (0,2) -> (0,1) ceil
        weights_matrix = jnp.where(weights_matrix > 1.0, 1.0, weights_matrix) ** 2
        res = np_correlation_wrap(z0, z1, spearman=False, weights=weights_matrix, return_raw=return_raw)

        if return_raw:
            return pl.DataFrame(
                {
                    'cov': res[0],
                    'std1': res[1],
                    'std2': res[2],
                    'var_name': features
                }
            )
    # else:
    #     res = np_correlation_wrap(zs[0], zs[1], spearman=spearman)

    return pl.DataFrame(
        {
            'res': res,
            'var_name': features
        }
    )

def griddata_correlations(
        adatas: list[AnnData],
        griddata_keys: list[str],
        features: np.ndarray,
        *,
        ignore_zeros: bool = False,
        spearman: bool = True,
        signal_weighting: bool = False,
        return_raw: bool = False
) -> pl.DataFrame:
    
    if return_raw:
        assert signal_weighting, "Signal weighting needs to be true for return_raw"
    
    assert not (signal_weighting * spearman), "Both spearman and signal weighting cannot be both True"
    assert not (ignore_zeros * signal_weighting), "Both ignore_zeros and signal weighting cannot be both True"

    gds: list[Griddata] = [adata.uns['griddata'][k] for adata, k in zip(adatas, griddata_keys)]
    inter_shapes = np.array([np.intersect1d(gd["feature_names"], features).shape[0] for gd in gds])
    assert (inter_shapes == features.shape[0]).all(), "All features are not present in all the griddatas"

    ixs = [np.argsort(gd["feature_names"]) for gd in gds]
    imasks = [ix[np.searchsorted(gd["feature_names"], features, sorter = ix)] for ix, gd in zip(ixs, gds)]

    zs = [gd["z"][im] for im, gd in zip(imasks, gds)]
    zs = [z.reshape(z.shape[0], -1) for z in zs]

    return _correlation_internal(zs[0], zs[1], features, spearman, ignore_zeros, signal_weighting, return_raw)


def landscape_dynamics(
        adata: AnnData,
        griddata_key: str,
        categorical_gd_key: str,
        *,
        fdc: bool = True,
        spearman: bool = True
) -> pl.DataFrame:

    gd: Griddata = adata.uns['griddata'][griddata_key]
    gdc: Griddata = adata.uns['griddata'][categorical_gd_key]

    var_names = gd["feature_names"]
    uq_cats = [g for g in np.unique(gdc["z"]) if g != -1]
    catmasks = [gdc["z"] == uq for uq in uq_cats]
    validmask = gd["metadata"] > 0


    pshape = (len(catmasks), gd["z"].shape[0])
    res = np.zeros(pshape, dtype=np.float32)
    for i, cm in enumerate(catmasks):
        mask = validmask * cm
        values = gd["z"][:, mask]
        times = -gd["grid"][1][mask]

        res[i] = np_correlation_wrap(values, times, spearman=spearman)

    # res[np.isnan(res)] = 1
    # if fdc:
    #     res = sts.false_discovery_control(res.ravel(), method='bh').reshape(pshape)
    

    return pl.DataFrame(
        {
            'group': [uq_cats] * var_names.shape[0],
            'res': res.T,
            'var_name': var_names
        }
    ).explode(['group', 'res'])
