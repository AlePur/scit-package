import numpy as np
import scipy.sparse as sps
import scipy.stats as sts
from typing import Literal


def vectorized_chi2(
        group_sum: np.ndarray,
        group_n: np.int32 | np.ndarray,
        rest_sum: np.ndarray,
        rest_n: np.int32 | np.ndarray,
        *,
        rank_based_fc: bool = False,
        vector_on: Literal['values', 'groups'] = 'values',
        statistic: Literal['ratio', 'original'] = 'ratio'
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.ones(group_sum.shape, dtype=np.float64)
    pvals = scores.copy()

    # group sum is array of sum of values in group,
    # vector_on 'values' means constant group size
    for i in range(group_sum.shape[0]):
        if vector_on == 'groups':
            _group_n = group_n[i]
            _rest_n = rest_n[i]
        else:
            _group_n = group_n
            _rest_n = rest_n
        if group_sum[i] == 0 and rest_sum[i] == 0:
            continue
        table = np.array(
            [
                [group_sum[i], _group_n - group_sum[i]],
                [rest_sum[i], _rest_n - rest_sum[i]]
            ]
        )
        statres = sts.chi2_contingency(table)
        #oddsratio = sts.contingency.odds_ratio(table)

        if statistic == 'original':
            scores[i] = statres.statistic

        pvals[i] = statres.pvalue

    if statistic == 'ratio':
        scores = ((1 + group_sum) / group_n) / ((1 + rest_sum) / rest_n)
        if rank_based_fc:
            scores = (sts.rankdata(group_sum) / sts.rankdata(rest_sum))
        scores = np.log2(scores)
    return scores, pvals