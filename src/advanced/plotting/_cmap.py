
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np

_name_p = Literal['gray', 'with_color', 'bwr']

def draw_cbar(cmap, vminmax: tuple[float, float] = (0,1)):
    plt.imshow(np.zeros((2,100)))
    norm = Normalize(vmin=vminmax[0],vmax=vminmax[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), orientation='horizontal')
    return plt.gcf()

def custom_cmap(name: _name_p):
    segments_g = [
        (0.0, (1.0, 1.0, 1.0)),
        (0.2, (0.7, 0.7, 0.7)),
        (1.0, (0.10, 0.10, 0.10))
    ]
    segments_wc = [
        (0.0, (1.0, 1.0, 1.0)),
        (0.5, (0.7, 0.85, 0.7)),
        (1.0, (0.0, 0.60, 0.0))
    ]
    segments_bwg = [
        (0.0, (0.0, 0.0, 0.6)),
        (0.2, (0.0, 0.3, 0.7)),
        (0.5, (0.75, 0.75, 0.7)),
        (0.8, (0.8, 0.0, 0.0)),
        (1.0, (0.5, 0.1, 0.1))
    ]

    return  LinearSegmentedColormap.from_list('testCmap', segments_g) if \
            name == 'gray' else \
            LinearSegmentedColormap.from_list('testCmap', segments_bwg) if \
            name == 'bwr' else \
            LinearSegmentedColormap.from_list('testCmap', segments_wc) if \
            name == 'with_color' else None
