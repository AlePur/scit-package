import numpy as np
import scipy.sparse as sps
import os

from ...plotting.int import boxplot, heatmap
from ...plotting._helper import MplWrap
from ..._utils import ArgAssert


def _plot_box(data, labels, label_size: int = 7, color: str = 'red',yminmax = None, title: str | None = None, ylabel: str | None = None, show=True):
    if any([len(i) == 0 for i in data]):
        raise ValueError("Data to be plotted in boxplot has list length of 0")
    return boxplot(
        [i for i in data],
        labels,
        color=color,
        label_size=label_size,
        ylabel=ylabel,
        title=title,
        yminmax=yminmax,
        show=show
    )


def _plot_with_line(data, labels, lines, cmap, vminmax, show_labels: bool = True):
    f = heatmap(
        np.array(data),
        labels,
        square_matrix=False,
        cmap=cmap,
        vminmax=vminmax,
        show_ylabels=False,
        show_xlabels=show_labels,
        show=False
    )
    for l in lines:
        if l == data[0].shape[0]:
            continue
        f.axes[0].plot([-0.5,4.5], [l-0.25, l-0.25], c='black', linewidth=1)
    return f


def _save(f, i: int = 0, _dir: str | None = None):
    if _dir is None:
        return
    else:
        f.savefig(os.path.join(_dir, f'{i}.pdf'))

def summary_boxplot(
        data: dict[str, dict[str, sps.csr_matrix]],
        labels_ordered: list[str],
        colors: list[str],
        log_scale: bool = False,
        log_scalar: list[float] | None = None,
        ignore_zero: bool = True,
        relative_max_per_gene: bool = False,
        normalize_per_gene: bool = False,
        cut_on_max_whisker: bool = False,
        *,
        ylabel: str | None = None,
        text_size: int = 7,
        show: bool = True
):

    fs = []
    keys = list(data.keys())

    t = lambda i, x: x
    if log_scale:
        _sclr = log_scalar
        if log_scalar is None:
            _sclr = [1.0] * len(keys)
        t = lambda i, x: np.log1p(_sclr[i]*x)
    
    for i,k in enumerate(keys):

        means = np.array([t(i, data[k][l].mean(axis=0).A1) for l in labels_ordered])
        if ignore_zero:
            means = means[:,means.sum(axis=0) != 0]
        if relative_max_per_gene:
            means = means @ np.diag(1.0 / means.max(axis=0))
        elif normalize_per_gene:
            means = means @ np.diag(1.0 / means.sum(axis=0))

        if cut_on_max_whisker:
            iql = np.percentile(means, 25, axis=1)
            iqh = np.percentile(means, 75, axis=1)
            iqr = iqh - iql
            # np.percentile(means, cut_on_max_q, axis=1)
            yminmax = (-0.01, np.max(iqh+iqr*1.5) + 0.01)
        else:
            yminmax = None

        f = _plot_box(means, labels_ordered, text_size, colors[i],yminmax,title=k, ylabel=ylabel,show=show)
        
        # _save(f, i, save_dir)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            fs.append(f) 

    if not show:
        return fs


def summary_heatmap(
        data: dict[str, dict[str, sps.csr_matrix]],
        labels_ordered: list[str],
        memberships: np.ndarray,
        cmaps: list[str],
        *,
        # cocluster: bool = False,
        log_scale: bool = False,
        modality_weights: list[float],
        vminmaxs: list[tuple | None] | None = None,
        show_labels: bool = True,
        show: bool = True
):

    keys = list(data.keys())
    from .._summary import _sortable_data

    if vminmaxs is None:
        vminmaxs = [None] * len(keys)

    fs = []

    for i,k in enumerate(keys):
        data_summed = [data[k][l].mean(axis=0).A1 for l in labels_ordered]
        if log_scale:
            data_summed = [np.log1p(100*ds) for ds in data_summed]
        sort_data = _sortable_data(data, modality_weights, labels_ordered).mean(axis=1)
        order, lines = sort_by_membership(sort_data, memberships)
        f = _plot_with_line([d[order] for d in data_summed], labels_ordered, lines, cmaps[i], vminmaxs[i], show_labels=show_labels)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            fs.append(f) 

    if not show:
        return fs

def sort_by_membership(X, membership):
    X = np.asarray(X)
    membership = np.asarray(membership)
    lines = []
    
    # Initialize output arrays
    sorted_indices = np.empty(len(X), dtype=int)
    
    # Get unique membership groups
    unique_groups = np.unique(membership)
    
    pos = 0
    # Sort within each group
    for group in unique_groups:
        # Get indices for this group
        mask = membership == group
        lines.append(mask.sum())
        group_indices = np.where(mask)[0]
        
        # Sort values within this group
        group_values = X[mask]
        sort_order = np.argsort(group_values)
        
        # Place sorted indices in output
        n_group = len(group_indices)
        sorted_indices[pos:pos+n_group] = group_indices[sort_order]
        pos += n_group
    
    return sorted_indices, np.cumsum(lines)