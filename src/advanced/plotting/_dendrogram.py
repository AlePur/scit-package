import numpy as np
from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import rc_context

from ..._utils import ArgAssert
from ...plotting._helper import MplWrap

def community_dendrogram(
        adata: AnnData,
        fontsize: int = 10,
        *,
        padding: float = 0.5,
        additive: bool = True,
        log_scale: bool = True,
        show_labels: bool = True,
        show: bool = True
) -> None | Figure:
    """
    Parameters
    ----------
    fontsize
    log_scale
        Whether branch height is log scale
    """
    ArgAssert('community_tree' in adata.uns.keys(), "please run community_hierarchy before trying to plot the dendrogram")

    linkage = adata.uns['community_tree']['linkage'].copy()

    order = adata.uns['community_tree']['order']

    plw = MplWrap(show)

    x = list(np.argsort(order))
    y_level = [0] * order.shape[0]
    #plw.ax.scatter(x, y_level,s=10)
    with rc_context({'font.size': fontsize}):
        for _x, i in enumerate(order):
            plw.ax.text(_x, -padding, str(i), 
                        horizontalalignment='center',
                        verticalalignment='center')
    
    for i in range(linkage.shape[0]):
        _cs = (int(linkage[i][0]), int(linkage[i][1]))
        x.append((x[_cs[0]] + x[_cs[1]]) / 2)
        # lkg is the linkage strength
        lkg = linkage[i][2]
        lkg = np.log1p(lkg) if log_scale else lkg
        if additive:
            lkg += np.array([y_level[_cs[0]], y_level[_cs[1]]]).max()
        y_level.append(lkg)
        plw.ax.plot(
            [x[_cs[0]], x[_cs[0]], x[_cs[1]], x[_cs[1]]], 
            [y_level[_cs[0]], y_level[len(y_level)-1], y_level[len(y_level)-1], y_level[_cs[1]]],
            c='0'
        )
    plw.remove_axis()
    plw.remove_ticks()
    return plw.show()