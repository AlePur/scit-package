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