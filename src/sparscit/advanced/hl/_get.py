
from typing import Literal
import numpy as np

from .._cv2 import ImageToolsCV2
from ...plotting._helper import MplWrap
from .._griddata import Griddata
from ...advanced.plotting._2d import _summarybar

def transform(gd):
    gd.z = np.sign(gd.z) * np.power(10 * np.abs(gd.z), 1/1.3)

def get_cbar(adata):
    plw = MplWrap(False)
    _summarybar(adata, plw.ax)
    plw.remove_axis()
    plw.remove_ticks()
    cbar = ImageToolsCV2.crop_white(
        ImageToolsCV2.fig_to_img(plw.fig), 
        padding=0
    )
    return cbar
