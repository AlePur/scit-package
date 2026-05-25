import numpy as np
from anndata import AnnData
import polars as pl
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata
from typing import Literal

from .._summarydata import SummaryData
from ...plotting._helper import MplWrap
from ...tools._layers import LayerConfig
from ..._utils import ArgAssert
from .._griddata import Griddata, griddata_numerical, _get_griddata_z
from ...plotting._bar import comparative_barplot
from ..._utils import _get_memberships
from ...tools._layers import layer_pick_features

_pstyle = Literal['scatter', 'smooth', 'both']

from scipy.stats import binned_statistic


def bin_to_grid_scipy(x, y, grid_cell_size):
    """
    Bin y-values onto a regular grid using scipy.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    x_min = x.min()
    x_max = x.max()
    
    n_cells = int(np.ceil((x_max - x_min) / grid_cell_size))
    bins = np.linspace(x_min, x_min + n_cells * grid_cell_size, n_cells + 1)
    
    ys, bin_edges, _ = binned_statistic(x, y, statistic='mean', bins=bins)
    xs = bin_edges[:-1] + grid_cell_size / 2
    
    return ys, xs

def summary_overtime(
        summary: SummaryData,
        labels: list[str],
        colors: list[str],
        cell_index: int = 0,
        *,
        yminmax: tuple[float, float] | None = None,
        show_all_lines: bool = False,
        weak_line_style: dict = {
            'linewidth': 0.5,
            'alpha': 0.1
        },
        line_style: dict = {
            'linewidth': 2
        }
) -> None:
    line_style = line_style.copy()
    weak_line_style = weak_line_style.copy()

    uq = np.unique(summary.memberships)

    for _uq in uq:
        plw = MplWrap(True)
        m = summary.memberships == _uq
        for i in range(summary.dimensions):
            d = summary.data[m][:, summary.layer_width*i:(summary.layer_width*(i+1))]
            d = d[:, summary.timesteps*cell_index:summary.timesteps*(1+cell_index)]
            plw.ax.plot(np.arange(d.shape[1]), d.mean(axis=0), label=labels[i], color=colors[i], **line_style)
            if show_all_lines:
                for _d in d:
                    plw.ax.plot(np.arange(d.shape[1]), _d, color=colors[i], **weak_line_style)
            # plw.ax.plot(discrete, X, **lab)

        if yminmax is not None:
            plw.ax.set_ylim(yminmax[0], yminmax[1])
        plw.ax.set_xlabel('Time')
        plw.ax.set_ylabel('Signal')
        if labels is not None:
            plw.ax.legend()
        plw.despine()
        plw.show()


def time_correlation(
        adatas: list[AnnData],
        configs: list[LayerConfig],
        obs_key: str,
        cell_type: str,
        time_key: str,
        feature_name: str,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
        scalars: list[int] | None = None,
        *,
        rank_signal: bool = False,
        plot_style: _pstyle = 'scatter',
        gauss_sigma: float = 3,
        binsize: float = 1,
        plot_zeros: bool = True,
        yminmax: tuple[float, float] | None = None,
        scatter_style: dict = {
            's': 5
        },
        xlabels: list | None = None,
        line_style: dict = {
            'linewidth': 5
        },
        show: bool = False
) -> None:
    scatter_style = scatter_style.copy()
    line_style = line_style.copy()

    values = []
    times = []

    for adata, lc in zip(adatas, configs):

        cats, cat_names, uq_cats = _get_memberships(adata, obs_key)

        if cell_type not in cat_names:
            raise ValueError(f'{cell_type} not found')
        
        cat_ix = np.argmax(cell_type == cat_names)

        layer_pick_features(adata, lc, feature_name)
        values.append(lc.transform(adata, use_cached_mask=True)[cats == cat_ix][:,0].todense().A1)
        times.append(adata.obs[time_key].to_numpy()[cats == cat_ix])

    plw = MplWrap(show=show)

    for i,(x,y) in enumerate(zip(times, values)):

        lab = {}
        if labels is not None:
            lab['label'] = labels[i]
        if colors is not None:
            lab['c'] = colors[i]
        if scalars is not None:
            y = y*scalars[i]

        if plot_style == 'scatter' or plot_style == 'both':
            if not plot_zeros:
                _x = x[y!=0]
                _y = y[y!=0]
            else:
                _x = x
                _y = y
            # if rank_signal:
            #     plw.ax.scatter(_x, rankdata(_y), **lab, **scatter_style)
            # else:
            plw.ax.scatter(_x, _y, **lab, **scatter_style)
        if plot_style == 'smooth' or plot_style == 'both':
            if plot_style == 'both' and 'label' in lab:
                del lab['label']
            biny, binx = bin_to_grid_scipy(x, y, binsize)

            dy = gaussian_filter1d(biny, gauss_sigma)
            plw.ax.plot(binx, dy, **lab, **line_style)

    plw.ax.set_xlabel('Time')
    plw.ax.set_ylabel('Signal')
    if yminmax is not None:
        plw.ax.set_ylim(yminmax[0],yminmax[1])
    if labels is not None:
        plw.ax.legend()
    if xlabels is not None:
        plw.set_text_xlabels(xlabels, start_at_1=False)
    plw.despine()
    return plw.show()

def time_correlation_griddata(
        adata: AnnData,
        griddata_key: str,
        categorical_gd_key: str,
        obs_key: str,
        cell_type: str,
        feature_name: str,
        *,
        plot_style: _pstyle = 'scatter',
        gauss_sigma: float = 3,
        binarize: bool = False,
        scatter_style: dict = {
            's': 5
        },
        line_style: dict = {
            'linewidth': 5
        },
        min_cells: int = 10,
) -> None:
    scatter_style = scatter_style.copy()
    line_style = line_style.copy()

    gd: Griddata = adata.uns['griddata'][griddata_key]
    categorical_gd: Griddata = adata.uns['griddata'][categorical_gd_key]
    ArgAssert(categorical_gd["categorical"], 'categorical_gd has to be Categorical')

    uq_cats = np.unique(categorical_gd["z"])
    if cell_type not in adata.obs[obs_key].cat.categories:
        raise ValueError(f'{cell_type} not found')
    c = np.argmax(adata.obs[obs_key].cat.categories == cell_type)
    catmask = categorical_gd["z"] == c


    plw = MplWrap(True)

    # adata.obs['copied_data'] = lc.transform(adata, use_cached_mask=True, binarize=binarize).T.todense().A1
    validmask = gd["metadata"] > 0
    mask = validmask * catmask
    values = _get_griddata_z(gd, feature_name)[mask]
    times = -gd["grid"][1][mask]
    lab = {}
    #if labels is not None:
    #    lab['label'] = labels[i]
    #if colors is not None:
    #    lab['c'] = colors[i]

    if plot_style == 'scatter' or plot_style == 'both':
        plw.ax.scatter(times, values, **lab, **scatter_style)

    plw.ax.set_xlabel('Time')
    plw.ax.set_ylabel('Signal')
    #if labels is not None:
    #    plw.ax.legend()
    plw.despine()
    plw.show()
    #plw.ax.set_title(f'Correlation: {"{0:.2f}".format(corrs.statistic)}; -logpval: {"{0:.1f}".format(-np.log(corrs.pvalue))}')


def differential_dynamics(
        dynamics,
        *,
        activity_threshold: float = 10.0,
        lfc_threshold: float = 0.2,
        alpha: float = 0.05,
        active_only: bool = True,
        score_key: str = 'score',
        activity_key: str = 'base_mean',
        show: bool = True
):
    df = dynamics

    if active_only:
        ArgAssert(activity_key in df.columns, 'This dynamics df does not contain expression data')
        df = df.filter(pl.col(activity_key) > activity_threshold)

    res = {}
    for g in df['group'].unique():
        _df = df.filter(pl.col('group') == g)
        resdict = {
            'ns': 0,
            'up': 0,
            'down': 0
        }
        # sig = _df['significant'].to_numpy().astype(np.bool_)

        increasing = _df.with_columns(((pl.col(score_key) > lfc_threshold) &  (pl.col('pval') < alpha)).alias('a'))['a'].to_numpy()
        decreasing = _df.with_columns(((pl.col(score_key) < -lfc_threshold) & (pl.col('pval') < alpha)).alias('a'))['a'].to_numpy()
        resdict['ns'] = (((~increasing) * (~decreasing))).astype(np.int32).sum()
        resdict['up'] = (increasing).astype(np.int32).sum()
        resdict['down'] = (decreasing).astype(np.int32).sum()
        res[g] = resdict

    return comparative_barplot(res, ['up', 'down', 'ns'], colors=['tab:blue', 'tab:orange', 'tab:gray'], show=show)