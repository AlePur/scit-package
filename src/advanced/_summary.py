import numpy as np
from anndata import AnnData
import scipy.sparse as sps
from copy import deepcopy

from ..tools._metabin import membership_summary
from ..tools._layers import LayerConfig, layer_pick_features
from ._summarydata import SummaryData
from .._utils import ArgAssert, _most_frequent_uint
from ..plotting._helper import MplWrap
from ._griddata import griddata_categorical, Griddata
from .plotting._2d import _getgrid, RN
from .._logging import logging

from sklearn.cluster import KMeans, SpectralCoclustering
from sklearn.decomposition import PCA


def summarize_multiple_modalities(
        adatas: list[AnnData],
        layers: list[LayerConfig],
        features: np.ndarray,
        obs_key: str,
        *,
        norm_per_modality: bool = True,
        celltype_norm: bool = False,
        sqrt_transform: bool = False
) -> dict[str, dict[str, sps.csr_matrix]]:
    d = {}
    for _adata, lc in zip(adatas, layers):
        layer_pick_features(_adata, lc, features)
        l = {}
        if celltype_norm:
            l = dict(
                normalize_per_class_obs_key=obs_key, normalize_keep_sum=True
            )
        summ = membership_summary(_adata, lc, obs_key, use_cached_mask=True, **l)
        summsum = 0
        for x in summ.values():
            summsum += x.sum()
        for x in summ.values():
            if norm_per_modality:
                x.data = 1000000 * x.data / summsum
            if sqrt_transform:
                x.data = np.sqrt(x.data)
        # if norm_over_celltypes:
        #     means = np.array([x.mean(axis=0).A1 for x in summ.values()])
        #     for i,(k,x) in enumerate(summ.items()):
        #         summ[k] @= sps.diags(means.sum(axis=0) / means[i])

        d[lc.layer] = summ 
    
    return d

def _sortable_data(
        data: dict[str, dict[str, sps.csr_matrix]],
        modality_weights: list[float] | None = None,
        sorted_l: list[str] | None = None,
        *,
        return_structured: bool = False
):
    ## first axis becomes features. Everything else is combined into one mixed axis
    ## Exception is return_structured=True
    modalities = list(data.keys())
    if sorted_l is None:
        cats = list(data[modalities[0]].keys())
    else:
        cats = list(sorted_l)
    features_n = data[modalities[0]][cats[0]].shape[1]
    _data = np.zeros((len(cats), len(modalities), features_n))
    for i,k in enumerate(cats):
        # mean of cell type in modality
        _data[i] = np.array([data[m][k].mean(axis=0).A1 for m in modalities])
    if modality_weights is not None:
        _data = _data.transpose((2,0,1)) @ np.diag(np.array(modality_weights))
        return _data.reshape((features_n), -1)
    if return_structured:
        return _data.transpose((2,1,0))
    return _data.transpose((2,1,0)).reshape((features_n), -1)

def cocluster_summaries(
        data: dict[str, dict[str, sps.csr_matrix]],
        modality: str,
        log_transform: bool = False,
        # ordering_layer_weights: dict[str, float] | None = None,
        *,
        n_components: int = 2
) -> np.ndarray:
    n_cl = len(next(iter(data.values())).keys())
    data = deepcopy(data)
    keys = list(data.keys())
    for k in keys:
        if k != modality:
            del data[k]
    
    _data = _sortable_data(data)
    if log_transform:
        _data = np.log1p(100*_data)

    _cc = SpectralCoclustering(n_clusters=n_cl, random_state=0).fit(_data)
    _cols = _cc.column_labels_
    _rows = _cc.row_labels_

    ni = np.argsort(_cols)
    _new = np.asarray(_rows).copy()
    for i in np.arange(n_cl):
        _new[_rows == i] = ni[i]

    return _new


def cluster_summaries(
        data: dict[str, dict[str, sps.csr_matrix]],
        n_clusters: int = 3,
        log_transform: bool = False,
        # cluster_keys_separately: bool = False,
        # separate_on: str | None = None,
        # ordering_layer_weights: dict[str, float] | None = None,
        *,
        n_components: int = 2
) -> np.ndarray:

    #  n_mod = len(list(data.keys()))
    
    def _make_cluster(_data):

        _data = _sortable_data(_data)
        if log_transform:
            _data = np.log1p(100*_data)

        if _data.shape[0] < 10:
            # Too small for PCA
            return np.zeros((_data.shape[0],))

        _pca = PCA(n_components=n_components)
        X_d = _pca.fit_transform(_data)
        logging.info(_pca.explained_variance_)

        memberships = KMeans(n_clusters=n_clusters).fit_predict(X_d)

        return memberships
    
    #if cluster_keys_separately:
    #    if separate_on is None:
    #        raise ValueError('separate_on is None')
    #    keys = list(next(iter(data.values())).keys())
    #
    #    _data = _sortable_data({'default': data[separate_on]}, return_structured=True, sorted_l=keys)
    #    fmask = np.argmax(_data[:,0,:], axis=1)
    #
    #    memberships = np.zeros(fmask.shape[0],dtype=np.uint32)
    #
    #    for i in np.unique(fmask):
    #        _min = memberships.max()
    #        memberships[fmask == i] = _make_cluster(
    #            {_k:{__k:__v[:,fmask == i] for __k, __v in _v.items()} for _k, _v in data.items()}
    #        )
    #        memberships[fmask == i] += _min + 1
    #
    #    return memberships
    #else:
    return _make_cluster(data)

def make_color_summary(
        adata: AnnData,
        min_cells_x: int,
        griddata_key: str,
        *,
        min_cells_y: int | None = None,
        y_fraction: float = 1.0,
        make_plot: bool = True
) -> None:
    """
    Plot summary bar for categorical (mostly cell type)

    Parameters
    ----------
    obs_key
        Key for categorical in .obs
    min_cells_x
        If not 0, trim grid on both ends to remove areas of low cell density
    y_fraction
        Float in range (0 - 1] for how much of the landscape to consider for summary. Smaller number means only cells in leaf clusters are used for summary
    cutoff
        Minimum distance to a real datapoint for a grid-datapoint to be considered. Zero (default) means infinite distance
    make_plot
        If this is False, skip plotting 

    Returns
    -------
    `adata.uns['summary']`
        Summary image data
    `adata.uns['grid']`
        Trimmed grid if trim_grid = True
    """

    gd: Griddata = adata.uns['griddata'][griddata_key]
    plw = MplWrap(True)

    grid_x, grid_y = _getgrid(adata)
    resolution = adata.uns['grid']['resolution']
    ArgAssert(gd["categorical"], "Griddata has to be categorical")
    _z = gd["z"][:, :int(gd["z"].shape[0] * y_fraction)]

    if min_cells_x > 0 or min_cells_y is not None:
        if min_cells_y is None:
            min_cells_y = 0
        pass_t_x = gd["metadata"].sum(axis=1) > min_cells_x
        pass_t_y = gd["metadata"].sum(axis=0) > min_cells_y
        firstr_x, lastr_x = (np.argmax(pass_t_x), np.argmax(np.flip(pass_t_x)))
        firstr_y, lastr_y = (np.argmax(pass_t_y), np.argmax(np.flip(pass_t_y)))

        grid_x = grid_x[firstr_x:resolution[0] - lastr_x, :][:, firstr_y:resolution[1] - lastr_y]
        grid_y = grid_y[firstr_x:resolution[0] - lastr_x, :][:, firstr_y:resolution[1] - lastr_y]
        _z = _z[firstr_x:resolution[0] - lastr_x, :][:, firstr_y:resolution[1] - lastr_y]
        adata.uns['grid']['xy'] = [grid_x, grid_y]
        adata.uns['grid']['resolution'] = list(grid_x.shape)
        logging.info(f"[x] grid trimmed from {resolution[0]} to {grid_x.shape[0]}")
        logging.info(f"[y] grid trimmed from {resolution[1]} to {grid_y.shape[1]}")

    cell_cats_x = [_most_frequent_uint(__z[__z != -1]) for __z in _z]
    _X = gd["color_map"][cell_cats_x].reshape((1, grid_x.shape[0], 3))
    adata.uns['summary'] = _X

    if make_plot:
        plw.ax.imshow(
            np.repeat(_X, RN, axis=0),
            origin='lower'
        )

    plw.remove_ticks()
    plw.show()
