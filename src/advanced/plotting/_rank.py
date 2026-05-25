import numpy as np
import scipy.stats as sts
from ...plotting.int import scatter

def rank_plot(mean0, mean1, log_scale: bool = True, labels: list | None = None):
    if log_scale:
        mean0 = np.log10(0.001 + mean0)
        mean1 = np.log10(0.001 + mean1)
    scatter(
        [
            np.c_[mean0, sts.rankdata(mean0)],
            np.c_[mean1, sts.rankdata(mean1)]
        ],
        labels=labels#, yminmax=(-2,3)
    )