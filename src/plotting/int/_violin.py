import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..._utils import ArgAssert
from ...plotting._helper import MplWrap
from ...tools._statistics import StatisticResult

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def violin(
        data: list[np.ndarray],
        labels: list[str],
        *,
        color: str = 'tab:red',
        colors: list[str] | None = None,
        show: bool = True,
        title: str | None = None,
        show_mean: bool = False,
        yminmax: tuple[float, float] | None = None,
        stat: StatisticResult | None = None,
        ylabel: str | None = None,
        label_size: str | None = None
) -> Figure | None:
    """
    Create violin plot.

    Parameters
    ----------
    data
    labels
    show
    stat
        :class:`StatisticResult` to plot significance

    Returns
    -------

    """
    plw = MplWrap(show)
    if stat is not None:
        ArgAssert(stat.data_len == len(data), 'Statistic test has been '
                                              'performed on data of different length than provided')

    parts = plw.ax.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False
    )

    heights = np.array([d.max() for d in data])

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color if colors is None else colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    qdata = np.array([np.percentile(d, [25, 50, 75]) for d in data])
    whiskers = np.array(
        [
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, qdata[:,0], qdata[:,2])]
    )
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, (qdata[:,1]).shape[0] + 1)
    dkw = dict(
        marker='o', color='white', s=30, zorder=3
    )
    if show_mean:
        plw.ax.scatter(inds, np.array([np.mean(d) for d in data]), **dkw)
    else:
        plw.ax.scatter(inds, qdata[:, 1], **dkw)
    plw.ax.vlines(inds, qdata[:,0], qdata[:,2], color='k', linestyle='-', lw=5)
    plw.ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    if stat is not None:
        for comb, res in zip(stat.test_scheme, stat.res):
            plw.annotate_brackets(comb[0], comb[1], res, np.arange(len(data))+1, heights)

    if yminmax is not None:
        plw.ax.set_ylim(yminmax[0], yminmax[1])
    if title is not None:
        plw.ax.set_title(title)
    plw.set_text_xlabels(labels, text_size=label_size)
    plw.despine()

    if ylabel is not None:
        plw.ax.set_ylabel(ylabel)
    return plw.show()


def boxplot(
        data: list[np.ndarray],
        labels: list[str],
        *,
        color: str = '#D43F3A',
        title: str | None = None,
        label_size: int | None = None,
        stat: StatisticResult | None = None,
        yminmax: tuple[float, float] | None = None,
        write_n: bool = False,
        ylabel: str | None = None,
        xlabel: str | None = None,
        show: bool = True,
) -> Figure | None:
    """
    Create boxplot

    Parameters
    ----------
    data
    labels
    color
        Color for boxplot
    title
        Title of plot
    label_size
        Size of text
    show

    """
    assert all([(len(d) > 0) for d in data]), "Empty data array"


    plw = MplWrap(show)

    bp = plw.ax.boxplot(
        data,
        patch_artist=True,  # Fill boxes with color
        medianprops=dict(color="white", linewidth=1.5),  # White median lines
        flierprops=dict(marker='o', markerfacecolor=color, markersize=4),  # Outlier styling
        whiskerprops=dict(color='black', linewidth=1),  # Whisker styling
        capprops=dict(color='black', linewidth=1),  # Cap styling
    )

    for box in bp['boxes']:
        box.set_facecolor(color)
        box.set_edgecolor('black')
        box.set_alpha(1)

    qdata = np.array([np.percentile(d, [25, 50, 75]) for d in data])
    whiskers = np.array(
        [
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, qdata[:, 0], qdata[:, 2])]
    )

    inds = np.arange(1, len(data) + 1)
    #plw.ax.scatter(inds, qdata[:, 1], marker='o', color='white', s=30, zorder=3)

    if stat is not None:
        heights = np.array([d.max() for d in data])
        for comb, res in zip(stat.test_scheme, stat.res):
            plw.annotate_brackets(comb[0], comb[1], res, np.arange(len(data))+1, heights)

    if write_n:
        lens = np.array([f"n={len(d)}" for d in data])
        plw.annotate_x(lens)

    if title:
        plw.ax.set_title(title)
    if yminmax is not None:
        plw.ax.set_ylim(yminmax[0], yminmax[1])
    if xlabel is not None:
        plw.ax.set_xlabel(xlabel)
    if ylabel is not None:
        plw.ax.set_ylabel(ylabel)

    plw.set_text_xlabels(labels, text_size=label_size)
    plw.despine()
    return plw.show()