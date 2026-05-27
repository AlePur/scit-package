import numpy as np
from .._helper import MplWrap


def barplot(
        values: np.ndarray,
        names: np.ndarray,
        *,
        color: str = 'tab:red',
        title: str | None = None,
        xlabel: str | None = None,
        h: bool = False,
        show: bool = True
) -> None:
    """Create a bar plot from values and labels.

    Parameters
    ----------
    values
        Heights (or widths for horizontal bars) of the bars
    names
        Category labels for each bar
    color
        Bar colour
    title
        Optional plot title
    xlabel
        Label for the value axis; defaults to ``'-ln(p_corr)'``
    h
        If ``True``, draw horizontal bars
    show
        Whether to display the plot immediately

    Returns
    -------
    Matplotlib Figure if ``show`` is ``False``, otherwise ``None``
    """
    plw = MplWrap(show)

    if h:
        plw.ax.barh(
            names, width=values, color=color
        )
    else:
        plw.ax.bar(
            names, height=values, color=color
        )
    if title is not None:
        plw.ax.set_title(title)

    deflab = '-ln(p_corr)'
    if xlabel is not None:
        deflab = xlabel
    if not h:
        plw.ax.tick_params(axis='x', rotation=90)
        plw.ax.set_xlabel(deflab)
    else:
        plw.ax.set_ylabel(deflab)
    plw.despine()
    return plw.show()