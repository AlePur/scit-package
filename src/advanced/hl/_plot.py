import matplotlib.pyplot as plt
from anndata import AnnData
import numpy as np

from ..plotting._2d import landscape2d, contour2d
from ..plotting._3d import landscape3d
from .._cv2 import ImageToolsCV2
from ..._settings import set_defaults

def plot_with_cbar(func, kwarg, cbar, path: str):
    """
    Provide function with kwarg to generate plot with added cbar
    """
    import cv2
    SCAL = 2
    set_defaults(figsize=(8*SCAL,6*SCAL))
    figure = func(**kwarg)

    landplot = ImageToolsCV2.fig_to_img(figure)
    result = ImageToolsCV2.overlay_images(
        landplot.copy(), 
        cv2.resize(cbar.copy(), (520*SCAL, 20*SCAL), cv2.INTER_NEAREST), 
        (170*SCAL,550*SCAL)
    )
    #cv2.putText(result, f'Gene: {gn}', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0))
    cv2.imwrite(path, result)