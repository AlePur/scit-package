import numpy as np
from .._logging import logging
from typing import Literal, Any

from ._hmm_data import _corr
from .._utils._kwargs import _format_kwargs
from .._utils import ArgAssert

_supported_dist = Literal['gamma', 'exponential']


class HMMListWrapper:

    def __init__(self, wrappers, cell_type_scheme: list[tuple[int, ...]], name: str):
        self.wrappers = wrappers
        self.name = name
        self.cell_type_scheme = [np.array(sc) for sc in cell_type_scheme]
        self.scheme = wrappers[0].scheme
        self.n_states = len(wrappers[0].state_dists)


class HMMWrapper:

    def __init__(self, model, scheme: list[str], independent: bool, name: str):
        self.model = model
        self.name = name
        self.independent = independent
        self.scheme = scheme
        self.state_dists = self.get_dist_state_variables()
        self.expected_observations = None
        self.observation_variance = None
        self.n_states = len(self.state_dists)

    def update_fit(self):
        self.state_dists = self.get_dist_state_variables()
        # This works for gamma and exponential
        self.expected_observations = np.array([np.prod(np.array(p), axis=1) for p in self.state_dists])
        sp = [np.array(p) for p in self.state_dists]
        for p in sp:
            p[:, -1] = p[:, -1]**2
        self.observation_variance = np.array([np.prod(np.array(p), axis=1) for p in sp])

    def get_dist_state_variables(self):

        params = []

        def _gamma(_shape, _rate):
            return [_shape, 1.0 / _rate]

        def _exp(_scale):
            return [_scale]

        for d in self.model.distributions:
            dpar = []
            if self.independent:
                dists = d.distributions
                for i in range(len(dists)):
                    if dists[i].name == 'Exponential':
                        dpar.append(_exp(dists[i].scales[0]))

                    else:
                        dpar.append(_gamma(dists[i].shapes[0], dists[i].rates[0]))  # .distributions[1])
            else:
                for i in range(len(d.shapes)):
                    dpar.append(_gamma(d.shapes[i], d.rates[i]))  # .distributions[1])
            params.append(dpar)

        return params


def create_hmm_system(
        n_states: int,
        hmm_scheme: list[_supported_dist],
        cell_type_scheme: list[tuple[int, ...]],
        hmm_name: str,
        *,
        inertia: float = 0.01,
        early_stop_tol: float = 1,
        pomegranate_densehmm_kwargs: dict = {}
) -> HMMListWrapper:
    """Create a system of HMMs, one per cell-type scheme entry.

    Each HMM in the system shares the same ``n_states`` and ``hmm_scheme``
    but is trained on a different subset of cell types defined by
    ``cell_type_scheme``.

    Parameters
    ----------
    n_states
        Number of hidden states per HMM
    hmm_scheme
        List of distribution names (``'gamma'`` or ``'exponential'``)
        per modality
    cell_type_scheme
        List of tuples; each tuple specifies which cell-type indices
        the corresponding HMM covers
    hmm_name
        Name identifier for the HMM system
    inertia
        Inertia parameter passed to pomegranate's DenseHMM
    early_stop_tol
        Early-stopping tolerance for training
    pomegranate_densehmm_kwargs
        Extra keyword arguments forwarded to ``pomegranate.hmm.DenseHMM``

    Returns
    -------
    An :class:`HMMListWrapper` containing one HMM per scheme entry
    """
    st_scheme_len = len(cell_type_scheme)
    return HMMListWrapper(
        [
            create_hmm(
                n_states=n_states,
                hmm_scheme=hmm_scheme,
                inertia=inertia,
                early_stop_tol=early_stop_tol,
                pomegranate_densehmm_kwargs=pomegranate_densehmm_kwargs
            )
            for i in range(st_scheme_len)
        ],
        cell_type_scheme,
        hmm_name
    )


def create_hmm(
        n_states: int,
        hmm_scheme: list[_supported_dist],
        hmm_name: str = 'unnamed',
        *,
        inertia: float = 0.01,
        early_stop_tol: float = 1,
        # init_states_from_wrapper: HMMWrapper | None = None,
        pomegranate_densehmm_kwargs: dict = {}
) -> HMMWrapper:
    """Create a single Hidden Markov Model using pomegranate.

    Builds a DenseHMM with the specified number of states and emission
    distributions.  Distributions can be independent (one per modality)
    or joint, depending on whether all entries in ``hmm_scheme`` are
    identical.

    Parameters
    ----------
    n_states
        Number of hidden states
    hmm_scheme
        List of distribution names (``'gamma'`` or ``'exponential'``)
        per modality
    hmm_name
        Name identifier for the model
    inertia
        Inertia parameter for training
    early_stop_tol
        Early-stopping tolerance
    pomegranate_densehmm_kwargs
        Extra keyword arguments forwarded to ``pomegranate.hmm.DenseHMM``

    Returns
    -------
    An :class:`HMMWrapper` around the created model
    """
    import pomegranate.hmm as phmm
    import pomegranate.distributions as pdist

    independent = ~(np.array([hs == hmm_scheme[0] for hs in hmm_scheme]).all())
    pomegranate_densehmm_kwargs = pomegranate_densehmm_kwargs.copy()
    n_modalities = len(hmm_scheme)

    dist_init = np.random.random((n_states, n_modalities)) * 2
    logging.info(dist_init)
    s = 0.5

    dist_list = list(
        iter(
            (
                (
                    pdist.Gamma(dist_init[i], np.array([s] * n_modalities))
                    if not independent else
                    pdist.IndependentComponents(
                       [
                           pdist.Gamma([dist_init[i][j]], [s])
                           if sc == 'gamma' else
                           pdist.Exponential([dist_init[i][j]])
                           for j, sc in enumerate(hmm_scheme)
                       ]
                    )
                ) for i in range(n_states)
            )
        )
    )

    tmatrix = np.full((n_states, n_states), 1, dtype=np.float32)
    starts = np.full((n_states,), 1, dtype=np.float32)
    starts = starts / starts.sum()
    tmatrix = tmatrix / tmatrix.sum(axis=0)

    kwarg = _format_kwargs(
        pomegranate_densehmm_kwargs,
        {
            "verbose":  True,
            "max_iter":  1000,
            "tol":  early_stop_tol,
            "inertia":  inertia,
        }
    )

    model = phmm.DenseHMM(
        dist_list,
        tmatrix,
        starts=starts,
        **kwarg
    )
    return HMMWrapper(
        model,
        hmm_scheme,
        independent,
        hmm_name
    )


def fit_hmm(
        hmm: HMMWrapper | HMMListWrapper,
        data, #HMMDATA
        data_layers: list[str],
        *,
        shuffled: bool = False,
        data_sample_start: int = 0,
        data_sample_end: int = -1,
        pretrain_global: bool = True
):
    """Fit an HMM (or HMM system) on the given data layers.

    When an :class:`HMMListWrapper` is passed, each sub-model is trained
    on its corresponding cell-type scheme.  If ``pretrain_global`` is
    ``True``, a global model is trained first and its parameters are
    used to initialise every sub-model.

    Parameters
    ----------
    hmm
        A single :class:`HMMWrapper` or an :class:`HMMListWrapper`
    data
        HMMData object containing the training observations
    data_layers
        Layer names to use as input features
    shuffled
        Whether to shuffle the data before training
    data_sample_start
        Start index for data sub-sampling
    data_sample_end
        End index for data sub-sampling (``-1`` = all)
    pretrain_global
        Pre-train a global model and copy its weights to all sub-models
        (only applicable when ``hmm`` is an :class:`HMMListWrapper`)
    """

    is_list = False
    if isinstance(hmm, HMMListWrapper):
        is_list = True

    def _fit(_hmm, _data: np.ndarray, scheme: np.ndarray | None = None):
        try:
            logging.info(f'Training data size: {data_sample_end-data_sample_start}')
            data_for_training = _data[data_sample_start:data_sample_end, :, :, :] \
                if scheme is None else _data[data_sample_start:data_sample_end, scheme, :, :]
            _hmm.model.fit(data_for_training)
        except KeyboardInterrupt:
            logging.info('Stopping training...')
        _hmm.update_fit()

    _X = data.get_data(data_layers, shuffled=shuffled)

    if not is_list:
        _fit(hmm, _X)
    else:
        if pretrain_global:
            logging.info('[-------------------]')
            logging.info(f'Starting global pre-training')
            from copy import deepcopy
            ghmm = deepcopy(hmm.wrappers[0])
            _fit(ghmm, _X)
            for i in range(len(hmm.wrappers)):
                hmm.wrappers[i] = deepcopy(ghmm)
        i = 0
        for sc in hmm.cell_type_scheme:
            logging.info('[-------------------]')
            logging.info(f'Starting training for model {i}')
            logging.info(f'With scheme {sc}')
            _fit(hmm.wrappers[i], _X, sc)
            i += 1


def calculate_correlation(
        hmm: HMMWrapper | HMMListWrapper,
        data, #: HMMData,
        data_layers: list[str],
        original_data_name: str,
        reconstructed_data_name: str,
        *,
        spearman: bool = True
) -> np.ndarray:
    """Calculate correlation between original and reconstructed data.

    Parameters
    ----------
    hmm
        Fitted HMM model (single or list wrapper)
    data
        HMMData object with reconstructed data stored in supplementary
    data_layers
        List of layer names
    original_data_name
        Name of the original data layer to correlate against
    reconstructed_data_name
        Name of the reconstructed data in supplementary storage
    spearman
        Use Spearman correlation (Pearson not yet implemented)

    Returns
    -------
    Array of correlation values
    """
    rd = data.supp[hmm.name].reconstructed_data[reconstructed_data_name]

    if not spearman:
        raise NotImplementedError()
    mp = np.argmax(np.array(data_layers) == original_data_name)
    return _corr([(rd[:, :, :, mp], data.get_slice(original_data_name))])
