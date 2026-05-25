import polars as pl
from polars import DataFrame
from anndata import AnnData
from pandas import DataFrame as pDF
import numpy as np

from .._logging import logging
from .._utils import ArgAssert


class Reference:

    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Feature reference with {str(self.df.shape[0])} features"

    def __init__(self, df: DataFrame) -> None:
        self.df = df
        pass


def symbol2id(
        ref: Reference,
        genes
) -> np.ndarray:
    n2i = dict(zip(ref.df['id'], ref.df['extra']))
    return np.array([n2i.get(p) for p in genes])

def get_gene_index(
        feature_names: np.ndarray,
        feature: str,
        error_on_not_found: bool = True
):
    """
    Get index of gene in feature names

    Parameters
    ----------
    feature_names
        Names of all features
    feature
        Name of feature to search for

    Returns
    -------

    """
    am = (feature_names == feature).argmax()
    if am == 0:
        if feature_names[0] != feature:
            if error_on_not_found:
                raise ValueError('No such feature found')
            else:
                return -1
    return am


def add_rna_var_metadata(
        adata: AnnData,
        ref: Reference,
        *,
        add_promoter_info: bool = True
) -> None:
    """
    Add gene information to an adata that has gene symbols as adata.var index

    Parameters
    ----------
    adata
    ref
        :class:`Reference`
    add_promoter_info
        Whether to add info about promoter location
    """
    idlist = ref.df['id'].to_numpy()
    isin = np.array([(_r in idlist) for _r in adata.var.index])
    if isin.sum() < adata.shape[1] * 0.1:
        logging.warning(
            "Very few genes were found in reference. This might mean your anndata does not have the correct var.index or"
            " the reference file (loaded gtf) is not suitable"
        )

    adata.var['chr'] = ['' if i else None for i in isin]
    adata.var['start'] = np.zeros((adata.shape[1],), dtype=np.uint32)
    adata.var['end'] = np.zeros((adata.shape[1],), dtype=np.uint32)

    _ix = adata.var.index[isin]
    _sortedids = np.argsort(idlist)

    mp = np.searchsorted(idlist, _ix, sorter=_sortedids, side='left')
    mp = _sortedids[mp]

    for n in ['chr', 'start', 'end']:
        adata.var.loc[_ix, n] = ref.df[n][mp].to_numpy()

    if add_promoter_info:
        adata.var['promoter'] = np.zeros((adata.shape[1],), dtype=np.uint32)
        adata.var.loc[_ix, 'promoter'] = np.choose(
            ref.df['strand'][mp] == "+",
            [ref.df['end'][mp].to_numpy(), ref.df['start'][mp].to_numpy()]
        )


def region_query(
        adata: AnnData,
        ref: Reference,
        chr: str,
        start: int,
        end: int
) -> pDF:
    """
    Find genes in range

    Parameters
    ----------
    adata
    ref
    chr
        Chromosome
    start
        Start of region
    end
        End of region

    Returns
    -------
    :class:`pandas.DataFrame` with genes

    """

    res = ref.df.filter(
        pl.col('chr') == chr,
        pl.col('start') - end < 0,
        pl.col('end') - start > 0
    )
    # logging.info(str(res))
    # [((cvar['start'] - res['end'].item()) < 0) * ((cvar['end'] - res['start'].item()) > 0)]

    return res.to_pandas()


def gene_query(
        adata: AnnData,
        ref: Reference,
        name: str,
        skip_info: bool = False
) -> pDF:
    """
    Find position of gene in .var using reference

    Parameters
    ----------
    adata
    ref
    name
        Name of feature
    skip_info
        Don't display logging info

    Returns
    -------
    :class:`pandas.DataFrame` from adata.var of bins overlapping with gene

    """

    ArgAssert('chr' in adata.var.keys(), "Please run toolkit.tl.add_metadata first")

    res = ref.df.filter(pl.col('id') == name)
    if not skip_info:
        logging.info(str(res))

    res = res[0]

    if (res.shape[0] != 0):
        cvar = adata.var[adata.var.chr == res['chr'].item()]
        return cvar[((cvar['start'] - res['end'].item()) < 0) * ((cvar['end'] - res['start'].item()) > 0)]
    else:
        return adata.var[np.full((adata.shape[1],), False)]
