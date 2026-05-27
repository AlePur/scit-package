from copy import deepcopy
import numpy as np
from anndata import AnnData
from typing import Literal, Any
import warnings

from ._pomegranate import HMMWrapper, HMMListWrapper
from ._hmm_data import HMMSuppData


def _get_expected(
        hmm: HMMWrapper,
        _y_proba: np.ndarray,
        get_var: bool = False
):
    _ex = hmm.expected_observations
    if get_var:
        _ex = hmm.observation_variance

    _shape = (_y_proba.shape[0], _y_proba.shape[1], _y_proba.shape[2])
    probabs = _y_proba.reshape(-1, hmm.n_states)

    return (probabs @ _ex).reshape(_shape[0], _shape[1], _shape[2], len(hmm.scheme))


def _predict(model, test_d, scheme) -> Any:
    import torch
    mask = np.zeros(test_d.shape, dtype=np.bool_)

    for jm in scheme:
        mask[:, :, :, jm] = True
    masked_data = torch.masked.masked_tensor(torch.tensor(test_d), torch.tensor(mask))
    return (
        np.array(
            [
                model.predict_proba(masked_data[i, :, :, :]).numpy()
                for i in range(masked_data.shape[0])
            ]
        )
    )


def predict_observations(
        hmm: HMMWrapper | HMMListWrapper,
        data, #, : HMMData,
        y_pred_name: str,
        data_name: str,
        *,
        predict_var: bool = False,
):
    """Reconstruct observations from HMM state probabilities.

    Uses the stored probabilities to compute expected (or variance) values
    per observation and stores the result in the data's supplementary
    storage.

    Parameters
    ----------
    hmm
        Fitted HMM model (single or list wrapper)
    data
        HMMData object with probabilities already computed
    y_pred_name
        Key for the probability array in ``data.supp``
    data_name
        Key under which to store the reconstructed data
    predict_var
        If ``True``, predict variance instead of expected value
    """
    _y_proba = data.supp[hmm.name].probabilities[y_pred_name]
    if isinstance(hmm, HMMListWrapper):
        expecteds = np.zeros((_y_proba.shape[0], _y_proba.shape[1], _y_proba.shape[2], len(hmm.scheme)), dtype=np.float32)
        for i, sc in enumerate(hmm.cell_type_scheme):
            expecteds[:, sc, :, :] = _get_expected(hmm.wrappers[i], _y_proba[:, sc, :, :], get_var=predict_var)
    else:
        expecteds = _get_expected(hmm, _y_proba, get_var=predict_var)

    s = HMMSuppData(
        hmm.name
    )
    s.reconstructed_data[data_name] = expecteds
    data.supp_add_data(s)


def get_probabilities(
        hmm: HMMWrapper | HMMListWrapper,
        data, #: HMMData,
        data_layers: list[str],
        scheme: list[int],
        data_name: str
):
    """Compute state probabilities for each observation using a fitted HMM.

    Runs the HMM forward algorithm on the specified data layers and stores
    the resulting probability arrays in the data's supplementary storage.

    Parameters
    ----------
    hmm
        Fitted HMM model (single or list wrapper)
    data
        HMMData object containing the observation data
    data_layers
        List of layer names to use as input
    scheme
        Cell-type scheme mapping for list wrappers
    data_name
        Key under which to store the computed probabilities
    """
    warnings.filterwarnings("ignore")
    _X = data.get_data(data_layers)

    if isinstance(hmm, HMMListWrapper):
        _shape = np.array(_X.shape)
        _shape[-1] = hmm.n_states
        y_proba = np.zeros(_shape, dtype=np.float32)

        for i, sc in enumerate(hmm.cell_type_scheme):
            test_d = _X[:, sc, :, :]
            probas = _predict(hmm.wrappers[i].model, test_d, scheme)
            y_proba[:, sc, :, :] = probas
    else:
        y_proba = _predict(hmm.model, _X, scheme)

    warnings.resetwarnings()
    s = HMMSuppData(
        hmm.name
    )
    s.probabilities[data_name] = y_proba
    data.supp_add_data(s)
