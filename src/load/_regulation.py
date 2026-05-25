import numpy as np
import polars as pl
import pandas as pd
from anndata import AnnData

from .._utils import ArgAssert
from .._logging import logging
from ..tools._regulation import RegInference


def regulatory_links(
        src: str,
        adata: AnnData,
        *,
        promoter: bool = True,
        promoter_size: int = 500
) -> RegInference:
    """
    Load links (from scGlue).
    Ambiguous promoters are given all feature names delimited by the string '||alt||'

    Parameters
    ----------
    src
    adata
        AnnData that contains the features in the first three (1 - 3) columns of the links file
    promoter
        Whether links are to gene promoters
    promoter_size
        Promoter size (default 500). This must be the same as used in generating the inference using scGlue

    Returns
    -------
    :class:`RegInference`
    """
    ArgAssert(promoter, "Non-promoter reg link parsing is not supported")
    df = pd.read_csv(
        src, delimiter="\t", names=['chr', 'start', 'end', 'chr2', 'start2', 'end2', 'score', 'pval', 'qval']
    ).dropna()
    for n in ['start', 'end', 'start2', 'end2']:
        df[n] = df[n].astype(np.int32)

    cmp_columns = ['chr', 'start', 'end']

    modv = adata.var[['chr', 'promoter']].copy()
    modv['start'] = np.clip(modv['promoter'] - promoter_size, 0, None)
    modv['end'] = modv['promoter'] + promoter_size
    modv = modv.reset_index(names="indexname")

    unique_df = df.reset_index(drop=True).groupby(by=cmp_columns).first().reset_index()

    mdf = pd.merge(unique_df, modv, on=cmp_columns, how='inner')
    diff = unique_df.shape[0] - mdf.groupby(by=cmp_columns).first().shape[0]

    if diff != 0:
        logging.warning(f"{diff} features listed in the inference were not found in the adata")

    dup = mdf.duplicated(cmp_columns).sum()
    if dup > 0:
        logging.info(f"{dup} features had promoter overlapping with at least one other feature, creating ambiguity")

    _dict = mdf.groupby(by=cmp_columns).agg({'indexname': lambda x: list(x)})[['indexname']].to_dict()['indexname']

    df['name'] = [_dict.get((v['chr'], v['start'], v['end'])) for _, v in df.iterrows()]

    df = pl.DataFrame(df).with_columns(
        (pl.col('chr2') + ":" + pl.col('start2').cast(pl.String) + "-" + pl.col('end2').cast(pl.String)).alias(
            'binname'
        )
    )
    # df = df.group_by(pl.col('name'))#.agg(pl.col('binname'))
    df = df.explode(pl.col('name'))  # [['name', 'score', 'binname']]
    df = df.sort('score').unique(['binname', 'name'], keep='last')
    df = df.filter(pl.col('score') > 0.1)

    return RegInference(df.to_pandas())