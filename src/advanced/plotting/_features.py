import numpy as np
from anndata import AnnData

from ...plotting._helper import MplWrap

def venn_active_overlap(
        activities: list[np.ndarray],
        *,
        labels: list[str] | None = None,
        show: bool = True,
        return_raw: bool = False
) -> None:
    import matplotlib_venn as mpv

    if len(activities) == 3:
        # (100, 010, 110, 001, 101, 011, 111)
        ss = (
                (activities[0] * (~activities[1]) * (~activities[2])).sum(),  # 100
                ((~activities[0]) * activities[1] * (~activities[2])).sum(),  # 010
                (activities[0] * activities[1] * (~activities[2])).sum(),     # 110
                ((~activities[0]) * (~activities[1]) * activities[2]).sum(),     # 001
                (activities[0] * (~activities[1]) * activities[2]).sum(),     # 101
                ((~activities[0]) * activities[1] * activities[2]).sum(),     # 011
                (activities[0] * activities[1] * activities[2]).sum(),         # 111
        )
        f = mpv.venn3
        if labels is None:
            labels = ['1', '2', '3']
    elif len(activities) == 2:
        # (10, 01, 11)
        ss = (
            (activities[0] * (~activities[1])).sum(),  # 10
            ((~activities[0]) * activities[1]).sum(),  # 01
            (activities[0] * activities[1]).sum(),  # 11
        )
        f = mpv.venn2
        if labels is None:
            labels = ['1', '2']
    else:
        raise ValueError(f'This number ({len(activities)}) of features is not supported for venn diagram drawing')

    if return_raw:
        return ss
    else:
        plw = MplWrap(show)
        f(subsets=ss, set_labels=labels, ax=plw.ax)
    return plw.show()