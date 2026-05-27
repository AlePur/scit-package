from anndata import AnnData
import anndata as ad
import numpy as np
from typing import Literal, Any, Self
from scipy.sparse import csr_matrix, issparse, diags, eye
import scipy.sparse as sps
import scipy.stats as sts
import pandas as pd

from ._transforms import _idf, _l2norm, _mfrob
from .._utils import ArgAssert
from .._logging import logging

mode_type = Literal['add', 'subtract']

def _format_var_names(fn) -> np.ndarray:
    """Convert feature names to a numpy array if not already."""
    if isinstance(fn, list):
        return np.array(fn)
    elif fn is not None:
        return np.array(list(fn))

class LayerConfig:
    """
    Configuration for a data layer in an AnnData object, specifying transformations
    and feature selection to apply when the layer is accessed.

    Stores normalization, transformation, and feature masking options. Use
    :func:`make_layer_config` to construct instances with defaults.

    Attributes
    ----------
    layer : str
        Key of the layer in ``adata.layers`` or ``adata.obsm``
    normalize_with_obs_counts : bool
        Whether to normalize by per-observation (cell) total counts
    log_transform : bool
        Whether to apply ``np.log1p`` transformation
    norm_total : float
        Scaling factor after count normalization
    feature_active_threshold : float | None
        Threshold for binarizing features as "active"
    in_obsm : bool
        If True, look up the layer in ``adata.obsm`` instead of ``adata.layers``
    feature_names : np.ndarray | None
        Names of all features in the layer
    normalize_with_l2_norm : bool
        Whether to normalize by L2 norm per observation
    idf_transform : bool
        Whether to apply inverse-document-frequency weighting
    mean_ranks_uniform_with : np.ndarray | None
        Reference mean vector for rank-based scalar alignment
    cache_transform : bool
        Whether to cache the transformed matrix (currently not fully implemented)
    scalar : float
        Multiplicative scalar applied to the transformed matrix
    cached_mask : np.ndarray | None
        Boolean mask of selected features
    """

    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Layer config: \"{self.layer}\" with normalize_with_obs_counts={self.normalize_with_obs_counts}, log_transform={self.log_transform}, norm_total={self.norm_total}, feature_active_threshold={self.feature_active_threshold}"

    def __init__(
            self,
            layer: str,
            normalize_with_obs_counts: bool,
            log_transform: bool,
            norm_total: float,
            feature_active_threshold: float | None,
            in_obsm: bool,
            feature_names: np.ndarray | None,
            cache_transform: bool,
            normalize_with_l2_norm: bool,
            mean_ranks_uniform_with: np.ndarray | None,
            idf_transform: bool
    ) -> None:
        self.scalar: float = 1.0
        self.in_obsm = in_obsm
        self.layer = layer
        self.normalize_with_obs_counts = normalize_with_obs_counts
        self.normalize_with_l2_norm = normalize_with_l2_norm
        self.idf_transform = idf_transform
        self.log_transform = log_transform
        self.norm_total = norm_total
        self.feature_active_threshold = feature_active_threshold
        self.feature_names = feature_names
        self.mean_ranks_uniform_with = mean_ranks_uniform_with
        try:
            _a = self.feature_names.to_numpy()
            self.feature_names = _a
        except Exception:
            pass
        self.cached_mask = None
        self.cache_transform = cache_transform
        self.cached_shape = None
        self.cached_matrix = None

    def get_feature_names(self, use_cached_mask: bool = True) -> np.ndarray:
        # This will always use cached_mask
        if self.feature_names is None:
            raise ValueError("feature_names were not provided when the layer config was created")
        if use_cached_mask == False:
            return self.feature_names.copy()
        return self.feature_names.copy() if self.cached_mask is None else self.feature_names[self.cached_mask].copy()

    def get_cached_mask(self) -> np.ndarray:
        if self.cached_mask is None:
            raise ValueError("cached_mask is not set. Please pick features")
        return self.cached_mask.copy()

    def get_threshold(self) -> float:
        if self.feature_active_threshold is None:
            raise ValueError(
                "This LayerConfig has not been provided a feature_active_threshold. Please generate a new one and set "
                "a relevant value"
            )
        return self.feature_active_threshold

    def get_shape(self, adata: AnnData) -> tuple[int, int]:
        self.verify_adata(adata)
        if self.in_obsm:
            return adata.obsm[self.layer].shape
        else:
            return adata.shape

    def verify_adata(self, adata: AnnData):
        ArgAssert(
            (self.layer in adata.obsm.keys()) if self.in_obsm else (self.layer in adata.layers.keys()),
            f"layer {self.layer} not found in anndata{'.obsm' if self.in_obsm else '.layers'}"
        )

    def transform(
            self,
            adata: AnnData,
            *,
            return_view_only: bool = False,
            use_cached_mask: bool = False,
            binarize: bool = False,
            raw_view: bool = False
    ) -> csr_matrix:
        self.verify_adata(adata)

        def _prepare_return(X: csr_matrix):
            if binarize:
                X.data = X.data > self.get_threshold()
            return X.tocsr()

        if self.cached_matrix is not None and self.cached_shape == adata.shape:
            # Will cause bug if layerconfig parameters are later modified
            raise NotImplementedError('cache is not implemented')
            view = self.cached_matrix
            # Already transformed
            return_view_only = True
        else:
            if self.in_obsm:
                view = adata.obsm[self.layer]
            else:
                view = adata.layers[self.layer]

        if use_cached_mask:
            ArgAssert(self.cached_mask is not None, "Please pick features with layer_pick_features")
            desiredmask = self.cached_mask
        else:
            desiredmask = np.ones((view.shape[1],), dtype=np.bool_)

        if raw_view:
            return view

        if return_view_only:
            return _prepare_return(self.scalar * view[:, desiredmask].copy())

        copy_now_cut_later = False
        if self.normalize_with_l2_norm:
            copy_now_cut_later = True
        if self.mean_ranks_uniform_with is not None:
            copy_now_cut_later = True

        # For obs_norm
        d = view.sum(axis=1).A1
        
        if copy_now_cut_later:
            _X: csr_matrix = view.copy()
        else:
            _X: csr_matrix = view[:, desiredmask].copy()
        if sps.issparse(_X):
            _X = _X.tocsr()
        else:
            _X = sps.csr_matrix(_X)

        #ArgAssert(not (self.log_transform and self.idf_transform), 'Both idf and log-transform cannot be True')

        # Be careful, some features might have been removed if copy_now_cut_later==False
        if self.idf_transform:
            _X = _idf(_X, allow_zeros=True)
        if self.normalize_with_obs_counts:
            if self.idf_transform:
                logging.warning('idf_transform should probably be used together with normalize_with_l2_norm')
            _X = (diags(1 / d) @ _X)
            _X.data *= self.norm_total
        if self.normalize_with_l2_norm:
            if self.normalize_with_obs_counts:
                raise ValueError('normalize_with_obs_counts and normalize_with_l2_norm cannot be used together')
            _X = _l2norm(_X.tocsr())
            _X.data *= self.norm_total
        if self.log_transform:
            _X.data = np.log1p(_X.data)

        # if self.cache_transform:
        #     if _X.data.nbytes > 2*(10**9):
        #         logging.warning('Caching the transformed matrix of this layer_config takes more than 2GB of memory. If '
        #                         'you want to keep your memory free, set cache_transform to false')
        #     self.cached_matrix = _X.copy()
        #     self.cached_shape = adata.shape

        if self.mean_ranks_uniform_with is not None:
            if self.idf_transform:
                raise ValueError('idf_transform and mean_ranks_uniform_with cannot be used together')
            _mean0 = self.mean_ranks_uniform_with
            assert isinstance(_mean0, np.ndarray), "mean_ranks_uniform_with needs to be a numpy array"
            _mean1 = get_layer_feature_means(raw_X = _X)
            sclrs = get_rank_scalars(_mean0, _mean1)
            _X = _X @ sps.diags(sclrs)
        
        if copy_now_cut_later:
            __X = self.scalar * _X[:, desiredmask]
        else:
            __X = self.scalar * _X
        return _prepare_return(__X)


def get_layer_feature_means(
        adata: AnnData | None = None,
        layer: LayerConfig | None = None,
        *,
        raw_X: Any | None = None,
        return_nans: bool = True
):
    """
    Compute the per-feature mean across all observations.

    Either provide ``raw_X`` directly, or provide both ``adata`` and ``layer``
    to transform the data first.

    Parameters
    ----------
    adata
        Annotated data matrix (required if ``raw_X`` is not given)
    layer
        Layer configuration (required if ``raw_X`` is not given)
    raw_X
        Pre-transformed sparse matrix; mutually exclusive with ``adata``/``layer``
    return_nans
        If True, include features with NaN means in the result;
        if False, filter them out

    Returns
    -------
    np.ndarray
        Array of per-feature means
    """
    assert (raw_X is None) or (adata is None and layer is None), "provide raw_X or adata+layer, not both"
    if raw_X is not None:
        _X = raw_X
    else:
        _X = layer.transform(adata, use_cached_mask=False)
    
    _m = _X.sum(axis=0).A1 / _X.shape[0]

    if return_nans:
        return _m
    return _m[~np.isnan(_m)]

def layer_pick_features(
        
        adata: AnnData,
        layer: LayerConfig,
        feature_index: str | int | np.ndarray | list[Any] | None = None,
        after_binarization: bool = False,
        filter_min: float | None = None,
        filter_max: float | None = None,
        *,
        verbose: bool = True
) -> None:
    """
    Set picked features in layer config to be used later.
    Does NOT guarantee that two layers with same picked features will have them in the same order, since
    the cached mask is a binary mask and does not change the order of the features

    Parameters
    ----------
    adata
    layer
        LayerConfig
    feature_index
        Indices or names of features to pick
    after_binarization
        Whether to binarize counts and use the result for thresholding
    filter_min, filter_max
        If this is set, pick all features passing threshold instead of by index

    """
    _nf = "It seems that there is no feature with the index as provided in var_index"
    _se = "Either min_threshold, max_threshold or feature_index has to be set"

    do_filter = filter_min is not None or filter_max is not None

    if not do_filter:

        ArgAssert(feature_index is not None, _se)
        if isinstance(feature_index, int) or isinstance(feature_index, str):
            feature_index = np.array([feature_index])
        elif isinstance(feature_index, list):
            feature_index = np.array(feature_index)

        if feature_index.dtype.kind == 'b':
            layer.cached_mask = feature_index
        elif feature_index.dtype.kind == 'S' or feature_index.dtype.kind == 'U' or feature_index.dtype.kind == 'O':
            if layer.feature_names is None:
                raise ValueError(
                    'feature_names is not set in the layer, so searching for feature'
                    ' based on name is not supported'
                )

            try:
                m = np.isin(layer.feature_names, feature_index)
            except Exception as e:
                logging.warning(_nf)
                raise e

            if m.sum() == 0:
                raise ValueError(_nf)

            layer.cached_mask = m
        else:
            layer.cached_mask = np.zeros((layer.get_shape(adata)[1],), dtype=np.bool_)
            layer.cached_mask[feature_index] = True
    else:
        data = layer.transform(adata)
        if after_binarization:
            data.data = data.data > layer.get_threshold()

        if filter_min is None:
            filter_min = 0
        if filter_max is None:
            filter_max = np.inf

        sumax = data.sum(axis=0).A1
        layer.cached_mask = (sumax > filter_min) * (sumax < filter_max)
        if verbose:
            logging.info(f'Picked {layer.cached_mask.sum()} features')

def make_layer_config(
        layer: str,
        normalize_with_obs_counts: bool = False,
        log_transform: bool = False,
        *,
        feature_names: np.ndarray | list[str] | None = None,
        in_obsm: bool = False,
        cache_transform: bool = False,
        norm_total: float = 10000.0,
        idf_transform: bool = False,
        normalize_with_l2_norm: bool = False,
        mean_ranks_uniform_with: np.ndarray | None = None,
        feature_active_threshold: float | None = None
) -> LayerConfig:
    """
    Make :class:`LayerConfig`. Different transformation options are provided,
    but not all options are mutually compatible. The default is to not transform the layer in any way.
    All transformations are lazy, which means the raw data in the AnnData object is never changed, and only transformed
    in memory when the need arises. The transformed layer is cached (not in the AnnData) if cache_transform is True.

    Parameters
    ----------
    layer
        Layers to consider
    normalize_with_obs_counts
        Normalize per-cell with obs counts
    log_transform
        Natural log transform the layer
    feature_names
        List of feature names for all features in layer
    in_obsm
        Whether layer is in .obsm. If False, try to find it in .layers
    cache_transform
        Whether to cache the transformed layer
    norm_total
        Number to scale counts by after normalizing
    idf_transform
        Use inverse-document-frequency to transform layer
    normalize_with_l2_norm
        Normalize per-cell by l2 norm
    feature_active_threshold
        Threshold (after all transformations) to consider count as "active"

    Returns
    -------
    :class:`LayerConfig`
    """
    if feature_names is None:
        logging.warn('No feature_names set for layer_config')

    return LayerConfig(
        layer,
        normalize_with_obs_counts,
        log_transform,
        norm_total,
        feature_active_threshold,
        in_obsm,
        _format_var_names(feature_names),
        cache_transform,
        normalize_with_l2_norm,
        mean_ranks_uniform_with,
        idf_transform
    )

def get_rank_scalars(mean0, mean1, eps=1e-2, scalar_minmax=(0.5, 2)):
    """
    Compute per-feature scaling factors so that the rank-ordered means of ``mean1``
    align with those of ``mean0``.

    Parameters
    ----------
    mean0 : np.ndarray
        Reference mean vector to align to
    mean1 : np.ndarray
        Mean vector to scale
    eps : float
        Minimum value to prevent division by zero; values below this are clamped
    scalar_minmax : tuple[float, float]
        Minimum and maximum values for clipping the resulting scalars

    Returns
    -------
    np.ndarray
        Per-feature scaling factors (clipped to ``scalar_minmax``)
    """
    mean0 = mean0.copy()
    mean1 = mean1.copy()
    assert mean0.shape[0] == mean1.shape[0]
    ranks0 = sts.rankdata(mean0)
    ranks1 = sts.rankdata(mean1)
    _ix = np.argsort(ranks1)
    placed = _ix[np.searchsorted(ranks1, ranks0, sorter=_ix)]
    if (mean0<eps).mean() > 0.1 or (mean1<eps).mean() > 0.1:
        print("Warning: rank scalar eps is too large, over 10{%} of the feature means are lower than eps.")
    # print(f"mean0: {(mean0<eps).sum()} near zero, mean1: {(mean1<eps).sum()} near zero, set those to {eps}")
    mean0[mean0<eps] = eps
    mean1[mean1<eps] = eps
    sclrs = mean0[placed] / mean1
    return np.clip(sclrs, scalar_minmax[0], scalar_minmax[1])

def copy_to_obs(
        adata: AnnData,
        layer_config: LayerConfig,
        *,
        binarize: bool = False
) -> None:
    """
    Add data from layer or .obsm to obs for plotting.

    Transforms the data according to the layer configuration, optionally
    binarizes it, then writes the per-observation sum into ``adata.obs['copied_data']``.

    Parameters
    ----------
    adata
        Annotated data matrix
    layer_config
        Layer configuration specifying which data to transform and how
    binarize
        Use threshold set in LayerConfig to make counts binary (passing or not passing)

    Returns
    -------
    None
        Writes the result into ``adata.obs['copied_data']``
    """

    X = layer_config.transform(adata, use_cached_mask=True)

    if binarize:
        th = layer_config.get_threshold()
        X.data = X.data > th

    adata.obs['copied_data'] = X.sum(axis=1).A1

def normalize_layers(
        adatas: list[AnnData],
        layer_configs: list[LayerConfig],
        sample_k: int,
        *,
        random_seed: int | None = None
):
    """
    Write layer scalars into LayerConfigs.

    Samples ``sample_k`` observations from each AnnData, computes the Frobenius
    norm of the residual (X X^T - I) for each layer, and sets each layer's
    ``scalar`` attribute so that layers are proportionally balanced.

    Parameters
    ----------
    adatas
        List of AnnData objects, one per layer
    layer_configs
        List of :class:`LayerConfig` objects corresponding to each AnnData
    sample_k
        Number of observations to sample for norm estimation
    random_seed
        Random seed for sampling observations

    Returns
    -------
    None
        Modifies ``layer_configs[i].scalar`` in place
    """
    gen = np.random.default_rng(seed=random_seed)

    X_norm = [l.transform(adatas[i]) for i,l in enumerate(layer_configs)]
    ws = np.zeros((len(X_norm)), dtype=np.float64)
    for i in range(len(X_norm)):
        X_sample = X_norm[i][gen.choice(adatas[i].shape[0], size=sample_k, replace=False), :]
        ws[i] = 1.0 / _mfrob((X_sample @ X_sample.T) - eye(sample_k))
        logging.info(f"Scale {i}th layer by {ws[i]}")

    for i, l in enumerate(layer_configs):
        l.scalar = np.sqrt(ws[i] / ws.sum())