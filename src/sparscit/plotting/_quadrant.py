import numpy as np
import scipy.stats as sts
from .int._scatter import scatter

def quad_text(ax, quadrant_texts):
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_mean = 0
    y_mean = 0
    
    l = {
        'fontsize': 15,
        'color': 'black',
        'bbox': {
            #'boxstyle': 'pad=0.3',
            'ec': 'black',
            'lw': 2,
            'alpha': 0
        },
        'horizontalalignment': 'center', 'verticalalignment': 'center'
    }
    # Quadrant 1 (Top-Right)
    # Midpoint between x_mean and xmax, and y_mean and ymax
    ax.text(x_mean + (xmax - x_mean) * 0.5, y_mean + (ymax - y_mean) * 0.5,
            quadrant_texts["Q1"], **l)
    
    # Quadrant 2 (Top-Left)
    # Midpoint between xmin and x_mean, and y_mean and ymax
    ax.text(xmin + (x_mean - xmin) * 0.5, y_mean + (ymax - y_mean) * 0.5,
            quadrant_texts["Q2"], **l)
    
    # Quadrant 3 (Bottom-Left)
    # Midpoint between xmin and x_mean, and ymin and y_mean
    ax.text(xmin + (x_mean - xmin) * 0.5, ymin + (y_mean - ymin) * 0.5,
            quadrant_texts["Q3"], **l)
    
    # Quadrant 4 (Bottom-Right)
    # Midpoint between x_mean and xmax, and ymin and y_mean
    ax.text(x_mean + (xmax - x_mean) * 0.5, ymin + (y_mean - ymin) * 0.5,
            quadrant_texts["Q4"], **l)

def plot_with_qtext(
        x1: np.ndarray,
        x2: np.ndarray,
        *,
        a: float = 0.5,
        s: int = 5,
        xlabel: str = "",
        ylabel: str = ""
):
    dta = np.c_[x1, x2]
    f=scatter(
        dta,
        alpha=a, s=s, show=False
    )
    tot = dta.shape[0]
    N_x_pos = (dta[:, 0] > 0).sum()
    N_x_neg = (dta[:, 0] < 0).sum()
    N_y_pos = (dta[:, 1] > 0).sum()
    N_y_neg = (dta[:, 1] < 0).sum()
    E_Q1 = (N_x_pos * N_y_pos) / tot**2
    E_Q2 = (N_x_neg * N_y_pos) / tot**2
    E_Q3 = (N_x_neg * N_y_neg) / tot**2
    E_Q4 = (N_x_pos * N_y_neg) / tot**2
    R_Q1 = ((dta[:,0] > 0) * (dta[:,1] > 0)).sum() / tot
    R_Q2 = ((dta[:,0] < 0) * (dta[:,1] > 0)).sum() / tot
    R_Q3 = ((dta[:,0] < 0) * (dta[:,1] < 0)).sum() / tot
    R_Q4 = ((dta[:,0] > 0) * (dta[:,1] < 0)).sum() / tot
    f.axes[0].set_xlabel(xlabel)
    f.axes[0].set_ylabel(ylabel)
    quad_text(
        f.axes[0],
        {
            "Q1": '{0:.3g}%\n(e:{1:.3g}%)'.format(100 * R_Q1, 100 * E_Q1),
            "Q2": '{0:.3g}%\n(e:{1:.3g}%)'.format(100 * R_Q2, 100 * E_Q2),
            "Q3": '{0:.3g}%\n(e:{1:.3g}%)'.format(100 * R_Q3, 100 * E_Q3),
            "Q4": '{0:.3g}%\n(e:{1:.3g}%)'.format(100 * R_Q4, 100 * E_Q4)
        }
    )

    r = sts.chisquare(np.array([R_Q1, R_Q2, R_Q3, R_Q4])*tot, f_exp=np.array([E_Q1, E_Q2, E_Q3, E_Q4])*tot)
    print(f"Pvalue: {r.pvalue}")
    return r