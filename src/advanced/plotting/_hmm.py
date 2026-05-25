import numpy as np

from .._pomegranate import HMMWrapper
from ...plotting.int import heatmap, histogram
from ..._logging import logging
from ...plotting._helper import MplWrap
from ...advanced.plotting._2d import landscape2d


def _empty_grid(
        shape: tuple[int, int]
):
    # Broken
    newgrid = np.zeros(shape, dtype=np.float32)
    _gd = Griddata(
        newgrid,
        newgrid,
        False,
        [newgrid.copy(), newgrid.copy()],
        skip_z_init=True
    )
    return _gd

def hmm_likelihoods(
    hmm: HMMWrapper,
    data,
    data_layers: list[str]
):
    # Broken
    _X = data.get_data(data_layers)
    likelihood = hmm.model.summarize(np.vstack(_X)).sum()
    logging.info(f"total likelihood: {likelihood}")
    logging.info(f"fraction positive: {(hmm.model.summarize(np.vstack(_X)) > 0).sum() / np.vstack(_X).shape[0]}")
    return histogram(hmm.model.summarize(np.vstack(_X)))


def transition_matrix(
        hmm: HMMWrapper,
        cmap: str = 'Blues_r',
        *,
        add_: int = 10
):
    w = -hmm.model.edges.numpy().copy()
    w[np.isinf(w)] = (w[~np.isinf(w)]).max() + add_
    heatmap(w, np.arange(hmm.n_states), cmap=cmap)


def _plot_state(
        dist_vars: list[list],
        scheme: list[str],
        title: str = "",
        *,
        xrange=(0, 10)
):
    from scipy.stats import gamma
    from scipy.stats import expon
    x = np.linspace(*xrange, 100)
    plw = MplWrap(True)
    plw.despine()
    plw.ax.set_title(title)

    def plot_gamma(name, _shape, _scale):
        # Plot the distribution
        pmf = gamma.pdf(x, _shape, scale=_scale)
        plw.ax.plot(
            x, pmf,
            # marker='o',
            # linestyle='--',
            label=f'{name}; a = {"{:.2f}".format(_shape)}; th = {"{:.2f}".format(_scale)}'
            )

    def plot_exp(name, _scale):
        # Plot the distribution
        pmf = expon.pdf(x, scale=_scale)
        plw.ax.plot(
            x, pmf,
            # marker='o',
            # linestyle='--',
            label=f'{name}; s = {"{:.2f}".format(_scale)}'
            )

    for i, dist_name in enumerate(scheme):
        if dist_name == 'exponential':
            plot_exp(dist_name, dist_vars[i][0])
        else:
            plot_gamma(dist_name, dist_vars[i][0], dist_vars[i][1])
    plw.ax.legend()


def hidden_states(
        hmm: HMMWrapper,
        xminmax: tuple[int, int]
):
    for i in range(hmm.n_states):
        _plot_state(hmm.state_dists[i], hmm.scheme, title=f'State {i}', xrange=xminmax)


def hmm_observations(
        data: np.ndarray,
        *,
        cmaps: list[str] = ('Blues', 'Reds', 'Greens')
):
    _shape = (data.shape[0], data.shape[1])

    rgd = _empty_grid(_shape)
    for i in range(data.shape[2]):
        rgd.z = data[:, :, i]  # [:,::-1]
        landscape2d(None, rgd, summary=False, cmap=cmaps[i])
