from anndata import AnnData
import numpy as np
from typing import Any
import numpy as np
import matplotlib.ticker as ticker
import textwrap

from ._helper import MplWrap
from .._utils import _get_memberships, ArgAssert

class CWheel:
    def __init__(self, states: list):
        self.state = -1
        self.cols = states

    def next(self):
        self.state += 1
        return self.cols[self.state % len(self.cols)]


def comparative_barplot(
        data: dict[str, Any],
        keys: list[str],
        *,
        width: float = 0.5,
        colors: list[str] | tuple = ('tab:red', 'tab:brown', 'tab:blue', 'tab:green'),
        show: bool = True
) -> None:

    ArgAssert(len(colors) == len(keys), "Lengths of keys and colors must be the same")
    ArgAssert(isinstance(data, dict), "Data must be given in a dict")
    uq_cats = np.sort(list(data.keys()))
    cwheel = CWheel(colors)
    plw = MplWrap(show=show)
    first = True

    for i, cat in enumerate(uq_cats):
        bottom = 0
        current_data = data[cat]
        if first:
            use_legend = True
            first = False
        else:
            use_legend = False
        for k in keys:
            arg = {
                "bottom": bottom,
                "color": cwheel.next()
            }
            if use_legend:
                arg["label"] = k
            p = plw.ax.bar(i, current_data[k], width, **arg)
            bottom += current_data[k]

    plw.set_text_xlabels(uq_cats, start_at_1=False)
    plw.ax.tick_params(axis='x', rotation=90)
    plw.despine()
    plw.fig.legend()
    return plw.show()


def smart_barplot(
        values: np.ndarray,
        names: np.ndarray,
        color_lambda: np.ndarray,
        colors: tuple,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ten_power: bool = False,
        text_size: float | None = None,
        wrap_width: int = 50,
        show: bool = True
) -> None:
    plw = MplWrap(show)
    kwarg = {
        'color': np.where(
            color_lambda, colors[0], colors[1]
        )
    }

    # 1. Use numerical positions instead of text for the bars
    positions = np.arange(len(values))

    # 2. Draw horizontal bars
    plw.ax.barh(
        positions, width=values, **kwarg
    )
    plw.ax.invert_yaxis()
    
    # 3. Hide standard y-ticks completely
    plw.ax.set_yticks([])
    
    # 4. Add text manually inside the bar
    # Add a slight 1% offset so the text isn't stuck to the spine
    x_offset = np.max(values) * 0.01 if len(values) > 0 else 0
    max_val = np.max(values) if len(values) > 0 else 1

    for val, y_pos, name in zip(values, positions, names):

        bar_ratio = val / max_val if max_val > 0 else 1
        
        # We enforce a hard floor (e.g., 10 chars) so that extremely short bars 
        # do not wrap every single letter into a massive unreadable vertical stack.
        dynamic_wrap_width = max(10, int(bar_ratio * wrap_width))
        wrapped_name = textwrap.fill(str(name), width=dynamic_wrap_width)

        plw.ax.text(
            x_offset, y_pos, wrapped_name, 
            ha='left', 
            va='center', 
            color='black',    # White contrasts best against tab:blue/tab:red
            fontsize=text_size,       # Adjust as necessary
            clip_on=True      # Prevents extremely long text from bleeding off the chart
        )

    if ten_power:
        def format_func(x, pos):
            if x == 0:
                return "0"
            return f"$10^{{-{x:.1f}}}$"

        plw.ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    if title is not None:
        plw.ax.set_title(title)
        
    if xlabel is not None:
        plw.ax.set_xlabel(xlabel) 
            
    plw.despine()
    return plw.show()

def barplot(
        values: np.ndarray,
        names: np.ndarray,
        color_lambda: np.ndarray,
        colors: tuple,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ten_power: bool = False,
        h: bool = False,
        text_size: float | None = None,
        show: bool = True
) -> None:
    plw = MplWrap(show)
    kwarg = {
        'color': np.where(
            color_lambda, colors[0], colors[1]
        )
    }

    if h:
        import matplotlib.ticker as ticker

        plw.ax.barh(
            names, width=values, **kwarg
        )
        plw.ax.invert_yaxis()
        if ten_power:
            def format_func(x, pos):
                if x == 0:
                    return "0"
                return f"$10^{{-{x:.1f}}}$"

            # 3. Apply the formatter
            plw.ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    else:
        plw.ax.bar(
            names, height=values, **kwarg
        )
        if ten_power:
            raise NotImplementedError()
    if title is not None:
        plw.ax.set_title(title)
    if not h:
        plw.ax.tick_params(axis='x', rotation=90)
    if xlabel is not None:
        plw.ax.set_ylabel(xlabel)
    # else:
    #     plw.ax.set_ylabel(deflab)
    plw.despine()
    return plw.show()