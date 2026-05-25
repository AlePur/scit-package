import numpy as np
from anndata import AnnData
from typing import Literal, Any, Self
import scipy.stats as sts

from .._logging import logging
from .._utils import ArgAssert

_bgt = Literal['bin', 'gene']
_transt = Literal['log1p', 'log1p_log1p', 'signed_log1p', 'none']


class HMMSuppData:
    def __init__(self, hmm_name: str):
        self.name = hmm_name
        self.probabilities = {}
        self.reconstructed_data = {}

    def merge(self, other: Self, warn_has_intersection: bool = True) -> Self:
        assert self.name == other.name, "Names of support data not matching"
        sp = np.array(list(iter(self.probabilities.keys())))
        op = np.array(list(iter(other.probabilities.keys())))
        srd = np.array(list(iter(self.reconstructed_data.keys())))
        ord = np.array(list(iter(other.reconstructed_data.keys())))
        int_o = np.intersect1d(sp, op)
        int_b = np.intersect1d(srd, ord)

        if not warn_has_intersection:
            assert int_o.shape[0] == 0
            assert int_b.shape[0] == 0
        else:
            for i in int_o:
                logging.warn(f"probability data '{i}' overwritten")
                #sp = sp[sp != i]
            for i in int_b:
                logging.warn(f"prediction data '{i}' overwritten")
                #srd = srd[srd != i]

        for i in op:
            self.probabilities[i] = other.probabilities[i]
        for i in ord:
            self.reconstructed_data[i] = other.reconstructed_data[i]
        return self

def _corr_ct(
        dcom: list[tuple[np.ndarray, np.ndarray]],
        lag: int = 0
):
    res_data_corr = []
    siz = dcom[0][0].shape[0]
    siz_ct = dcom[0][0].shape[1]
    siz_t = dcom[0][0].shape[2]

    for i, dc in enumerate(dcom):
        _dc0 = dc[0].reshape(siz, siz_ct, -1)
        _dc1 = dc[1].reshape(siz, siz_ct, -1)
        r = np.zeros((siz, siz_ct), np.float32)
        for j in range(siz):
            for k in range(siz_ct):
                r[j][k] = sts.spearmanr(_dc0[j][k][lag:], _dc1[j][k][:siz_t - lag])[0] \
                    if lag >= 0 else \
                    sts.spearmanr(_dc0[j][k][:siz_t + lag], _dc1[j][k][-lag:])[0]
        res_data_corr.append(r)
    return np.array(res_data_corr)

def _corr(
        dcom: list[tuple[np.ndarray, np.ndarray]],
        ignore_zero: bool = True,
        adjusted_q: float | None = 0.1
):
    res_data_corr = []
    siz = dcom[0][0].shape[0]
    rgn = np.random.default_rng()

    for i, dc in enumerate(dcom):
        _dc0 = dc[0].reshape(siz, -1).copy()
        _dc1 = dc[1].reshape(siz, -1).copy()
        if _dc0.min() >= 0 and _dc1.min() >= 0:
            all_values_over_zero = True
        else:
            all_values_over_zero = False
        r = np.zeros((siz,), np.float32)
        for j in range(siz):
            if ignore_zero and all_values_over_zero:
                m = ~((_dc0[j] == _dc0[j].min()) * (_dc1[j] == _dc1[j].min()))  #.astype(np.bool_)
            else:
                m = np.ones_like(_dc0[j], dtype=np.bool_)

            if adjusted_q is not None:
                ncells = m.sum()
                npick = ncells // 3
                res = []
                for k in range(30):
                    s = rgn.choice(np.arange(ncells), (npick,), replace=True)
                    prs = sts.pearsonr(
                        _dc0[j][m][s],
                        _dc1[j][m][s],
                    ).statistic
                    res.append(prs)
                prs = np.quantile(np.array(res), adjusted_q)
            else:
                # r[j] = sts.spearmanr(_dc0[j][m], _dc1[j][m]).statistic
                prs = sts.pearsonr(_dc0[j][m], _dc1[j][m]).statistic
                # kdl = sts.kendalltau(_dc0[j][m], _dc1[j][m]).statistic
            r[j] = prs
        res_data_corr.append(r)
    return np.array(res_data_corr)
