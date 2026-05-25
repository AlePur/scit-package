import scipy.stats as sts
import scipy.sparse as sps
from anndata import AnnData
import numpy as np
import polars as pl
from typing import Literal, Callable
from sklearn.preprocessing import StandardScaler
from scipy.stats import false_discovery_control
import jax.numpy as jnp
from functools import partial
from typing import Any
import jax
from jax import lax,  block_until_ready, jit

from ._layers import get_layer_feature_means
from ._metabin import membership_summary, _membership_summary_on_array
from ._optax import likelihood_ratio_test, fit_log_regression, LogisticParams
from ._basic_stats import vectorized_chi2
from ._cstats import mannwhitneyu_from_summary, ranks_from_sparse
from .._logging import PBar
from ..tools._layers import LayerConfig
from .._utils import _csr_mean, _get_memberships
from .._logging import logging
from .._utils import ArgAssert

_test_type = Literal['ttest', 'chisquare', 'mannwhitneyu', 'wilcoxon', 'likelihood']


class StatisticResult:
    def __init__(self, data_len: int, test_scheme: list, res: np.ndarray, score: np.ndarray | None = None):
        self.data_len = data_len
        self.test_scheme = test_scheme
        self.res = res
        self.score = score


class Markers:
    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Markers, {self.df.shape[0]} features (top markers {'not ' if self.top_markers is None else ''}determined)"

    def __init__(self, df: pl.DataFrame, names: np.ndarray, pop: np.ndarray):
        self.df = df
        self.names = names
        self.top_markers: list[pl.DataFrame | None] | None = None
        self.negative_markers = False
        self.population = pop
        self.alpha = 0.05


def likelihood_test_on_adata(
        adata: AnnData,
        cmpdata: AnnData,
        obs_key: str,
        lc: LayerConfig,
        *,
        log_transform_for_regression: bool = True,
        log_transform_covariate: bool = True,
        test_anti_categorical: bool = False,
        mean_ranks_normalize: bool = False,
        min_lfc: float = 0.25,
        min_base_mean: float | None = None
) -> pl.DataFrame:
    logging.info(f"Running likelihood test with minimum LFC={min_lfc}; min_base_mean={min_base_mean}")
    Xs: list[dict[str, Any]] = []
    Xs_with_log: list[dict[str, Any]] = []
    cvs: list[dict] = []
    f_names = lc.get_feature_names()
    has_logtrans = lc.log_transform
    if has_logtrans and (log_transform_covariate or log_transform_for_regression):
        raise ValueError("Log_transform is True in layerconfig, log_transform_for_regression and log_transform_covariate cannot be set to true (would log transform twice)")

    for i, _adata in enumerate([adata, cmpdata]):
        if mean_ranks_normalize:
            if i == 1:
                _means = get_layer_feature_means(adata, lc) # Always use first adata !
                lc.mean_ranks_uniform_with = _means
        if lc.in_obsm:
            cva = _adata.obsm[lc.layer].sum(axis=1).A1
        else:
            cva = _adata.layers[lc.layer].sum(axis=1).A1
        if log_transform_covariate:
            cva = np.log(cva)
        _anti = False
        if (i == 1) and test_anti_categorical:
            _anti = True
        cvs.append(_membership_summary_on_array(
            _adata, obs_key, cva, anti_summary=_anti
        ))
        xd = membership_summary(_adata, lc, obs_key, anti_summary=_anti)
        Xs.append(xd)
        if log_transform_for_regression:
            # Mutate 
            xdl = {k:v.copy() for k,v in xd.items()}
            for k in xdl.keys():
                xdl[k].data = np.log1p(xdl[k].data)
            Xs_with_log.append(xdl)
 
    lc.mean_ranks_uniform_with = None

    ks = set(list(Xs[0].keys()))
    assert ks == set(list(Xs[1].keys())), "obs_key must have same categoricals in both adatas"

    pvals = []
    scores = []
    keys = []
    basemeans = []
    for k in ks:
        X0 = Xs[0][k]
        X1 = Xs[1][k]
        cv0 = cvs[0][k]
        cv1 = cvs[1][k]
        data_X = sps.vstack((X0, X1))
        if log_transform_for_regression:
            X0 = Xs_with_log[0][k]
            X1 = Xs_with_log[1][k]
            data_reg = sps.vstack((X0, X1))
        pval, score, basemean = likelihood_test(
            data_X,
            np.concatenate((np.zeros_like(cv0), np.ones_like(cv1))).astype(np.bool_),
            covariate=np.concatenate((cv0, cv1)), min_lfc = min_lfc, min_base_mean=min_base_mean, 
            data_for_regression=data_reg if log_transform_for_regression else None
        )
        keys.append(np.full_like(pval, fill_value=k, dtype=np.dtypes.StringDType))
        pvals.append(pval)
        scores.append(score)
        basemeans.append(basemean)

    df=pl.DataFrame(
        {'pval': pvals, 'score': scores, 'name': [f_names] * len(pvals), "base_mean": basemeans, 'group': keys}
    ).explode(['pval', 'score', 'name', "base_mean", 'group'])
    return df


def likelihood_test(
        data: sps.csr_matrix,
        mask: np.ndarray,
        *,
        data_for_regression: sps.csr_matrix | None = None,
        covariate: np.ndarray | None = None,
        fdr: bool = True,
        min_lfc: float | None = None,
        min_base_mean: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uq = set(list(np.unique(mask)))
    assert uq == set([True, False]) or uq == set([0, 1]), 'Mask must only contain 0,1 or True,False'
    mask = mask.astype(np.bool_)

    #if covariate is None:
    #    covariate = np.log1p(data.sum(axis=1).A1)
    #else:
    es = f"Group 1 n={mask.sum()}, group 2 n={mask.shape[0] - mask.sum()}"
    assert covariate.shape[0] == data.shape[0]
    _mean1 = data[mask].mean(axis=0).A1
    _mean0 = data[~mask].mean(axis=0).A1
    filter_out = ((_mean1 == 0) + (_mean0 == 0)).astype(np.bool_)
    _mean0[filter_out] = 1
    _mean1[filter_out] = 1
    scores = np.log2(_mean1 / _mean0)
    base_means = (_mean1 + _mean0) / 2
    scores[filter_out] = 0
    base_means[filter_out] = 0
    _fmask = ~filter_out
    if min_base_mean is not None:
        _fmask = _fmask * (base_means > min_base_mean)
    if min_lfc is not None:
        _fmask = _fmask * (np.abs(scores) > min_lfc)

    if data_for_regression is not None:
        assert data.shape == data_for_regression.shape, "shapes of data and data_for_regression must be same"
        data = data_for_regression[:, _fmask]
    else:
        data = data[:, _fmask]
        
    logging.info(f'Running test on {_fmask.sum()} features. {es}')
    pvals = _likelihood_ratio_test_many_wrapper(data, covariate, mask)

    if fdr:
        pvals = false_discovery_control(pvals, method='bh')

    final_pvals = np.ones_like(scores)
    final_pvals[_fmask] = pvals

    return np.array(final_pvals), scores, base_means

def _likelihood_ratio_test_many_wrapper(
        X: sps.csr_matrix,
        z: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 1000
):
    """
    Parameters
    ----------
    X
        Data
    z
        Covariate
    Y
        Labels

    Returns
    -------
    P-values of X.shape[1] of whether adding X to the models improves the prediction.
    """

    assert np.isdtype(Y.dtype, np.bool_), 'Y needs to be boolean'
    X0 = z[:, None]
    X0_jax = jnp.array(X0)
    Y_jax = jnp.array(Y)
    X1_vector = jnp.array(np.array(X.todense())[:, :, None])

    return np.array(_likelihood_ratio_test_many(X0_jax, X1_vector, Y_jax, batch_size=batch_size))

@partial(jit, static_argnames=['batch_size', 'random_state'])
def _likelihood_ratio_test_many(
        X0: jnp.ndarray,
        X1: jnp.ndarray,
        Y: jnp.ndarray,
        batch_size: int,
        *,
        random_state: int = 0
) -> list[float]:
    print('Debug: traced _likelihood_ratio_test_many')
    key = jax.random.PRNGKey(random_state)
    init_s = LogisticParams(coef=jax.random.normal(key, (1,)) * 0.01, intercept=jnp.array(0.0))
    # likelihood_ratio_test_vec = vmap(likelihood_ratio_test, in_axes=(None, 1, None, None))
    # X1_vector = jnp.array(np.array(X.todense())[:,:,None])
    # result = likelihood_ratio_test_vec(X0_jax, X1_vector, Y_jax, warm_s)
    warm_s = fit_log_regression(X0, Y, warm_params=init_s, tol = 1e-6)

    def process_column(X1_col):
        jax.debug.print("Started batch", ordered=True)
        pvals = block_until_ready(likelihood_ratio_test(X0, X1_col, Y, warm_s))
        jax.debug.print("Finished batch", ordered=True)
        return pvals

    X1_transposed = jnp.transpose(X1, (1, 0, 2))
    return lax.map(process_column, X1_transposed, batch_size=batch_size)


def _interal_stat(
        data1: np.ndarray,
        data2: np.ndarray,
        test_type: _test_type,
        *,
        score_f: Callable | None = None,
        scipy_stat_kwargs: dict = {}
) -> tuple[np.ndarray, np.ndarray]:
    scipy_stat_kwargs = scipy_stat_kwargs.copy()
    _precomputed = None
    if score_f is not None:
        #_precomputed = np.log2(score_f(data1.sum(axis=0).A1) / cmpscore_f(data2.sum(axis=0).A1))
        _precomputed = score_f(data1.mean(axis=0).A1, data2.mean(axis=0).A1)

    if test_type == 'ttest':
        # scores, pvals = sts.ttest_ind_from_stats(
        #     mean1=_csr_mean(data1, 0),
        #     std1=np.sqrt(scalar_group.var_),
        #     nobs1=data1.shape[0],
        #     mean2=_csr_mean(data2, 0),
        #     std2=np.sqrt(scalar_rest.var_),
        #     nobs2=data2.shape[0],
        #     **scipy_stat_kwargs
        # )
        _test = sts.ttest_ind(data1, data2, **scipy_stat_kwargs)
        pvals = _test.pvalue
        scores = _test.statistic
        if _precomputed is not None:
            scores = _precomputed
        return pvals, scores
    elif test_type == 'wilcoxon':
        t = sts.wilcoxon(
            data1,
            data2,
            **scipy_stat_kwargs
        )
        scores = t.statistic
        if _precomputed is not None:
            scores = _precomputed
        return t.pvalue, scores
    elif test_type == 'mannwhitneyu':
        ArgAssert(
            (data1.dtype != np.bool_) * (data2.dtype != np.bool_),
            "Data can't be binary for mannwhitneyu"
        )
        scores, pvals = sts.mannwhitneyu(data1, data2, **scipy_stat_kwargs)
        if _precomputed is not None:
            scores = _precomputed
        return pvals, scores
    elif test_type == 'chisquare':
        ArgAssert(
            (data1.dtype == np.bool_) * (data2.dtype == np.bool_),
            'Data needs to be binary for chisquared test'
        )
        scores, pvals = vectorized_chi2(
            data1.sum(axis=0).A1,
            data1.shape[0],
            data2.sum(axis=0).A1,
            data2.shape[0]
        )
        if _precomputed is not None:
            scores = _precomputed
        return pvals, scores
    else:
        raise NotImplementedError(f"test '{test_type}' not found")

def statistic_test(
        data: list,
        test_scheme: list,
        *,
        test_type: _test_type = 'ttest',
        fdc: bool = False,
        welch: bool = False,
        scipy_stat_kwargs: dict = {},
        raw_output: bool = False
) -> StatisticResult | tuple[np.ndarray, np.ndarray]:
    """
    Create :class:´StatisticResult´ using one of supported statistical tests.

    Parameters
    ----------
    data
        List of numpy arrays, data from different groups to compare
    test_scheme
        Scheme of testing, list of tuples. Example: test_scheme=[(0,1)] means 0th and 1st data should be compared
    test_type
        One of 'ttest', 'chisquare', 'mannwhitneyu', 'wilcoxon'
    fdc
        Whether to control for false discovery rate using Benjamini-Hochberg
    welch
        If T-test, whether it should be Welch's T-test

    Returns
    -------
    :class:´StatisticResult´
    """
    res = []
    score = []

    _kwargs = scipy_stat_kwargs.copy()
    if welch:
        _kwargs['equal_var'] = False

    for ts in test_scheme:
        _res, _score = _interal_stat(np.array(data[ts[0]]), np.array(data[ts[1]]), test_type, scipy_stat_kwargs=_kwargs)
        res.append(_res)
        score.append(_score)

    res = np.array(res)
    res[np.isnan(res)] = 1
    if fdc:
        res = sts.false_discovery_control(res, method='bh')

    if raw_output:
        return res, np.array(score)
    return StatisticResult(len(data), test_scheme, res, score=np.array(score))


def _adata_test(
        adata: AnnData,
        data: sps.csr_matrix,
        obs_key: str,
        var_names: np.ndarray | list[str],
        *,
        test_type: _test_type = 'ttest',
        fdc: bool = True,
        rank_based_fc: bool = False,
        score_f: Callable | None = None,
        cmpadata: AnnData | None = None,
        cmpdata: sps.csr_matrix | None = None,
        covariates: np.ndarray | list[np.ndarray] | None = None
) -> pl.DataFrame:
    cats, cat_names, uq_cats = _get_memberships(adata, obs_key)
    if cmpadata is not None:
        cats_c, cat_names_c, _ = _get_memberships(cmpadata, obs_key)
        assert (cat_names_c == cat_names).all(), 'categories do not match in adata/cmpdata'
    res = []

    pdatas = []
    test_scheme = []
    with PBar.tqdm(total=uq_cats.shape[0]) as bar:
        for i, c in enumerate(uq_cats):
            mask = cats == c
            thisdata = data[mask]

            _covariates = None
            if cmpdata is None:
                restdata = data[~mask]
                if covariates is not None:
                    _covariates = [covariates[0][mask], covariates[0][~mask]]
            else:
                restdata = cmpdata[cats_c == c]
                if covariates is not None:
                    _covariates = [covariates[0][mask], covariates[1][cats_c == c]]

            pvals, scores = _interal_stat(
                thisdata, restdata, test_type, covariates=_covariates, rank_based_fc=rank_based_fc,
                score_f=score_f
            )

            res.append((pvals, scores))
            bar.n += 1
            bar.refresh()

    pvals = np.array([r[0] for r in res])
    scores = np.array([r[1] for r in res])
    if fdc:
        pvals = sts.false_discovery_control(pvals, method='bh')

    return pl.DataFrame(
        {
            'group': uq_cats,
            'res': pvals,
            'score': scores,
            'var_name': [var_names] * len(uq_cats)
        }
    ).explode(['res', 'score', 'var_name'])


def filter_top_markers(
        markers: Markers,
        n_top: int = 10,
        *,
        alpha: float = 0.05,
        keep_only_significant: bool = True,
        negative_markers: bool = False
) -> None:
    """
    Pick top markers for each category.

    Parameters
    ----------
    markers
        :class:`Markers`
    n_top
        Number of top markers
    alpha
        Maximum p-value
    keep_only_significant
        Purge unsignificant results
    negative_markers
        Find negative markers instead of positive ones

    """
    markers.alpha = alpha
    markers.negative_markers = negative_markers
    markers.top_markers = []

    uqs = markers.df['group'].unique().to_numpy()
    sclause = (pl.col('score') < 0) if negative_markers else (pl.col('score') > 0)

    for uq in uqs:
        _df = markers.df.filter(
            (pl.col('group') == uq)
            & sclause
        ).sort('res')
        if keep_only_significant:
            _df = _df.filter(pl.col('res') < alpha)
        _df = _df.top_k(n_top, by='res', reverse=True)
        pvals = _df['res'].to_numpy().copy()

        pm = pvals == 0
        if pm.sum() > 0:
            logging.info(f'[{markers.names[uq]}] Some ({pm.sum()}) p-values are extremely small (equal to zero)')

        pvals[pm] = 1
        lpvals = -np.log(pvals)
        markers.top_markers.append(
            (
                markers.names[uq],
                pl.DataFrame(
                    {
                        'score': _df['score'].to_numpy(),
                        'pval': _df['res'].to_numpy(),
                        'nlog_p': lpvals,
                        'name': _df['var_name'].to_numpy().copy()
                    }
                ) if lpvals.shape[0] > 0 else None
            )
        )


def enriched_in_group(
        adata: AnnData,
        layer: LayerConfig,
        obs_key: str,
        *,
        binarize: bool = False,
        likelihood_test: bool = False,
        fdc: bool = True
) -> Markers:
    """
    Calculate enriched features in layer, comparing each categorical to the rest.
    When binarize=True is passed, use chi squared test, otherwise use Mann-Whitney U.

    Parameters
    ----------
    adata
    layer
        LayerConfig
    obs_key
        Categorical key
    binarize
        Binarize layer
    fdc
        Whether to use false-discovery control (BH)

    Returns
    -------
    :class:`Markers`

    """
    _var_names = layer.get_feature_names()
    cats = adata.obs[obs_key].cat.codes.to_numpy()
    cat_names = adata.obs[obs_key].cat.categories.to_numpy()
    X = layer.transform(adata, binarize=binarize)

    test: _test_type
    if binarize:
        test = 'chisquare' if not likelihood_test else 'likelihood'
    else:
        test = 'mannwhitneyu'

    df = _adata_test(adata, X[:, layer.cached_mask], obs_key, _var_names, fdc=fdc, test_type=test, covariate=np.log1p(X.sum(axis=1).A1))
    return Markers(df, cat_names, _var_names)