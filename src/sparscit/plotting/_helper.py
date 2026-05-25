import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

from .._settings import settings
from .._utils._colors import hex_to_rgb
from numpy.typing import NDArray
from anndata import AnnData
from matplotlib.pyplot import Axes, Figure
from .._utils._colors import top100, top20
from typing import Any

class MplWrap:
    fig: Figure
    ax: Axes
    _show: bool

    def __init__(self, show: bool, figsize: tuple[int, int] | None = None, bind_fig: Figure | None = None, dummy: bool = False) -> None:
        self._show = show
        if figsize is not None:
            self.figsize = figsize
        else:
            self.figsize = settings.figsize
        self.constrained = settings.constrained
        plt.rcParams['figure.constrained_layout.use'] = self.constrained
        if bind_fig is None:
            self.create_axes()
        else:
            self.fig = bind_fig
            if not dummy:
                self.ax = self.fig.axes[0]

    def create_axes(self) -> None:
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = plt.subplot(111)

    def despine(
            self,
    ) -> None:
        self.ax.spines[['right', 'top']].set_visible(False)
    
    def annotate_x(self, data, y_pos_func=None, offset=0.05, fs=None, **kwargs):
        """
        Annotate each x-axis entry with a specific string (e.g., sample size 'n=10').

        :param data: list of strings to write above each x-tick. Length must match the number of xticks.
        :param y_pos_func: (Optional) A function that returns the y-coordinate for the text. 
                        It receives the index `i` of the x-position.
                        Default behavior: places text at the top of the current y-axis limit.
        :param offset: Vertical offset as a fraction of the y-axis range (0 to 1). 
                    Used if y_pos_func is None. Default is 0.05 (5%).
        :param fs: Font size.
        :param kwargs: Additional keyword arguments passed to ax.text (e.g., color, rotation).
        """
        
        # Get current axes (assuming self has an attribute .ax, otherwise use plt.gca())
        ax = getattr(self, 'ax', plt.gca())

        # Get x-axis positions currently plotted
        # For standard boxplots/barplots, these are usually 0, 1, 2... or 1, 2, 3...
        xticks = ax.get_xticks()
        
        # Validation
        if len(data) != len(xticks):
            raise ValueError(f"Length of data ({len(data)}) does not match number of x-ticks ({len(xticks)}).")

        # Get Y-axis limits to calculate offsets
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Default text styling
        text_kwargs = dict(ha='center', va='bottom', color='black')
        if fs is not None:
            text_kwargs['fontsize'] = fs
        text_kwargs.update(kwargs) # Merge user kwargs

        for i, text_str in enumerate(data):
            x_pos = xticks[i]
            
            # Determine Y position
            if y_pos_func is not None:
                # User provided a custom logic (e.g., getting the max of a specific dataset column)
                y_pos = y_pos_func(i)
            else:
                # Default: Place slightly above the current top visual limit (or just inside it)
                # You might want to adjust this logic depending on if you want it *on* the plot
                # or *above* the data. Here we place it relative to the axes top.
                y_pos = y_max + (offset * y_range)

            ax.text(x_pos, y_pos, str(text_str), **text_kwargs)


    def annotate_brackets(
            self,
            num1,
            num2,
            data,
            center,
            height,
            dh=.05,
            barh=.1,
            fs=None,
            maxasterix=4
        ):
        """
        Annotate barplot with p-values.

        :param num1: number of left bar to put bracket over
        :param num2: number of right bar to put bracket over
        :param data: string to write or number for generating asterixes
        :param center: centers of all bars (like plt.bar() input)
        :param height: heights of all bars (like plt.bar() input)
        :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
        :param fs: font size
        :param maxasterix: maximum number of asterixes to write (for very small p-values)
        """

        if type(data) is str:
            text = data
        else:
            text = ''
            p = .05

            while data < p:
                if text == '':
                    p *= 2

                text += '*'
                p /= 10.

                if maxasterix and len(text) == maxasterix:
                    break

            if len(text) == 0:
                text = 'n. s.'

        lx, ly = center[num1], height[num1]
        rx, ry = center[num2], height[num2]

        ax_y0, ax_y1 = plt.gca().get_ylim()
        dh *= (ax_y1 - ax_y0)
        barh *= (ax_y1 - ax_y0)

        y = max(ly, ry) + dh

        barx = [lx, lx, rx, rx]
        bary = [y, y + barh, y + barh, y]
        mid = ((lx + rx) / 2, y + barh)

        self.ax.plot(barx, bary, c='black')

        kwargs = dict(ha='center', va='bottom')
        if fs is not None:
            kwargs['fontsize'] = fs

        self.ax.text(*mid, text, **kwargs)

    def set_text_xlabels(self, labels: list[str], start_at_1: bool = True, text_size: int | None = None) -> None:
        self.ax.set_xticks(np.arange(len(labels)) + (1 * start_at_1), labels=labels)
        self.ax.set_xlim(0.25 - (1 * (not start_at_1)), len(labels) + 0.75 - (1 * (not start_at_1)))
        if text_size is not None:
            self.ax.tick_params(axis='x', which='major', labelsize=text_size)

    def set_text_ylabels(self, labels: list[str], start_at_1: bool = True, text_size: int | None = None) -> None:
        self.ax.set_yticks(np.arange(len(labels)) + (1 * start_at_1), labels=labels)
        self.ax.set_ylim(0.25 - (1 * (not start_at_1)), len(labels) + 0.75 - (1 * (not start_at_1)))
        if text_size is not None:
            self.ax.tick_params(axis='y', which='major', labelsize=text_size)

    def remove_axis(self) -> None:
        # removing the default axis on all sides:
        for side in ['bottom','right','top','left']:
            self.ax.spines[side].set_visible(False)

    def remove_ticks(
            self
    ) -> None:
        self.ax.set_xticks([],[])
        self.ax.set_yticks([],[])

    def show(self) -> Figure | None:
        if self._show:
            plt.show()
            plt.close(self.fig)
            return None
        return self.fig

    def arrowed_spines(self) -> None:

        xmin, xmax = self.ax.get_xlim() 
        ymin, ymax = self.ax.get_ylim()

        self.remove_axis()
        self.remove_ticks()

        dps = self.fig.dpi_scale_trans.inverted()
        bbox = self.ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height

        # manual arrowhead width and length
        hw = 1./30.*(ymax-ymin) 
        hl = 1./30.*(xmax-xmin)
        lw = 1. # axis line width
        ohg = 0. # arrow overhang

        # compute matching arrowhead length and width
        yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
        yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
        #diff = (xmax-xmin)*0.2

        # draw x and y axis
        self.ax.arrow(xmin, ymin, (xmax-xmin)*0.2, 0, fc='k', ec='k', lw = lw, 
                head_width=hw, head_length=hl, overhang = ohg, 
                length_includes_head= True, clip_on = False) 

        self.ax.arrow(xmin, ymin, 0, (ymax-ymin)*0.2, fc='k', ec='k', lw = lw, 
                head_width=yhw, head_length=yhl, overhang = ohg, 
                length_includes_head= True, clip_on = False)

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

class ColorUtil:
    color: str | None
    adata: AnnData

    def cls_init(
            self
        ) -> None:
        pass

    def __init__(
            self,
            adata: AnnData,
            color: str | None,
            explicit_series: Series | None = None
        ) -> None:

        self.adata = adata
        self.color = color
        self.explicit_series = explicit_series

        if (explicit_series is not None):
            self.__class__ = NumColorUtil
        else:
            if not (self.color is None):
                if self.adata.obs[self.color].dtype.name == "category":
                    self.__class__ = CategoricalColorUtil
                else:
                    self.__class__ = NumColorUtil
        self.cls_init()

    def get_col_arr(self) -> list[Any] | NDArray[Any]:
        return ['#777'] * self.adata.shape[0]

    def get_legend(self) -> list[Line2D]:
        return []


class NumColorUtil(ColorUtil):
    numarray: NDArray[Any]

    def cls_init(self) -> None:
        if self.explicit_series is not None:
            self.numarray = self.explicit_series.to_numpy().astype(np.int32)
        else:
            self.numarray = self.adata.obs[self.color].to_numpy()

    def get_col_arr(self) -> NDArray[Any]:
        return self.numarray

    def get_legend(self) -> list[Line2D]:
        handles: Any = []

        return handles

class CategoricalColorUtil(ColorUtil):
    all_named_colors: list
    colorlist: NDArray[np.object_] | NDArray[Any]
    _cats: list
    col_dict: dict[str, str]

    def cls_init(
            self
    ) -> None:

        self.all_named_colors = []

        self.categorical = True
        self._cats = list(self.adata.obs[self.color].dtype.categories)

        ncats = len(self._cats)

        self.all_named_colors.extend([_c for _c in mcolors.TABLEAU_COLORS.values()])
        if ncats < 30:
            self.all_named_colors.extend(top20)
        elif ncats < 110:
            self.all_named_colors.extend(top100)
        else:
            raise RuntimeError(
                "The plotting does not allow for more than 109 different categorical values. Please consider setting coax_numerical = True"
            )

        self.colorlist = np.array(self.all_named_colors)

        self.col_dict = self._get_col_dict()
        self.check_col_dict()

    def _get_col_dict(self, overwrite: bool = False) -> dict[str, str]:
        """
        Internal use only
        """
        seteq = False
        if 'colors_dict' in self.adata.uns.keys():
            seteq = set([str(c) for c in self._cats]) == set(list(iter(self.adata.uns['colors_dict'].keys())))
        if seteq is True and overwrite is False:
            colors_dict = self.adata.uns['colors_dict']
        else:
            _colors = [str(c) for c in self.colorlist[:len(self._cats)]]
            _scats = [str(c) for c in self._cats]
            colors_dict = dict(zip(_scats, _colors))
            self.adata.uns['colors_dict'] = colors_dict
        return colors_dict

    def check_col_dict(self) -> None:
        try:
            [self.col_dict[str(k)] for k in self._cats]
        except KeyError:
            self.col_dict = self._get_col_dict(True)

    def get_categorical_rgb(self) -> np.ndarray:

        X = np.zeros((len(self._cats), 3), dtype=np.int32)
        for i in range(len(self._cats)):
            X[i] = hex_to_rgb(self.col_dict[str(self._cats[i])])

        return X

    def get_col_arr(self) -> list[Any] | NDArray[Any]:

        col_arr = [self.col_dict[str(k)] for k in self.adata.obs[self.color]]
        return col_arr

    def get_legend(self) -> list[Line2D]:

        handles = []

        for c in self._cats:
            _c = str(c)
            handles.append(
                Line2D([0], [0], label=_c, marker='.', markersize=20, 
                       markerfacecolor=self.col_dict[_c], markeredgewidth=0, linestyle='')
            )

        return handles