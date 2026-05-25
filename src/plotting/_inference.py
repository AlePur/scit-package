import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from anndata import AnnData

from .._utils import ArgAssert
from ..tools._regulation import RegInference
from ._helper import MplWrap


def regulatory_links(
        reg: RegInference,
        name: str,
        *,
        zigzag: bool = True,
        text_size: float = 7.0,
        score_weight: int = 14,
        show: bool = True
) -> None:
    """
    Visualize regulatory links

    Parameters
    ----------
    reg
        Regulatory inference class
    name
        Name of feature to find
    zigzag
        Put text in a zig-zag fashion
    text_size
        Text font size
    score_weight
        How much link-score affects plotted link width
    """
    df = reg.df
    filtered_df = df[df['name'] == name]

    xmax = filtered_df.shape[0] - 1
    if (xmax == -1):
        #if (name in adata.var_names):
        #    raise ValueError("The feature exists but has no regulatory connections")
        #else:
        raise ValueError("The feature has no regulatory connections or doesn't exist")

    plw = MplWrap(show=show)
    _cmap = plt.get_cmap('Reds_r')
    qval_norm = mpl.colors.Normalize(vmin=np.log(0.0001), vmax=np.log(0.05)) #qval

    for i in range(xmax+1):
        el = filtered_df.iloc[i]
        color = _cmap(qval_norm(np.log(el['qval'])))

        f = '{0:.2g}'
        plw.ax.text(
            (i), -0.1, f"{el['chr2']}:{(el['start2'])}-{(el['end2'])}", 
            ha='center',
            va='center',
            fontsize=text_size
        )
        _xy1 = (i, 0)
        _xy2 = ((xmax / 2) + (i - xmax / 2)*0.1, 5)
        plw.ax.annotate(
            "",
            xy=_xy1, xycoords='data',
            xytext=_xy2, textcoords='data',
            arrowprops=dict(
                arrowstyle='->',color=color, alpha=1, 
                linewidth=score_weight*np.clip(el['score']-0.2, 0, None),
            )
        )
        plw.ax.text(
            (_xy1[0] + _xy2[0]) / 2, 
            2.5 + (-0.25 if (i%2 == 0) else 0.25) if zigzag else 0, 
            f"correlation: {f.format(el['score'])}\nqval: {f.format(el['qval'])}", 
            ha='center',
            va='center',
            fontsize=text_size
        )

    plw.ax.set(ylim=(-0.5, 5.5), xlim=(-0.5,xmax+0.5))
    t = filtered_df.iloc[0]
    plw.ax.set_title(f'Regulatory inference: {name}. Promoter at {t["chr"]}:{t["start"]}-{t["end"]}')
    plw.remove_axis()
    plw.remove_ticks()

    # Add colorbar
    #sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    #sm.set_array([])

    #cbar = plt.colorbar(sm)
    #cbar.set_label('Score')

    return plw.show()