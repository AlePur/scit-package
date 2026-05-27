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
    """
    Compute chi-squared test scores and p-values for each feature comparing a group against the rest.

    For each feature, constructs a 2×2 contingency table of (group_sum, group_n - group_sum)
    vs (rest_sum, rest_n - rest_sum) and computes the chi-squared statistic and p-value.

    Parameters
    ----------
    group_sum
        Array of summed values within the group for each feature
    group_n
        Number of observations in the group (scalar if ``vector_on='values'``,
        array if ``vector_on='groups'``)
    rest_sum
        Array of summed values in the rest for each feature
    rest_n
        Number of observations in the rest (scalar or array, same as ``group_n``)
    rank_based_fc
        If True, use rank-based fold change instead of ratio
    vector_on
        ``'values'`` means group_n/rest_n are scalar (constant group size across features);
        ``'groups'`` means they are arrays (varying group sizes)
    statistic
        ``'ratio'`` returns log2 fold-change as score; ``'original'`` returns the raw
        chi-squared statistic

    Returns
    -------
    scores : np.ndarray
        Feature scores (log2 fold-change or chi-squared statistic depending on ``statistic``)
    pvals : np.ndarray
        Chi-squared p-values for each feature
    """
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