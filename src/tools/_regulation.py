from anndata import AnnData
import pandas as pd
import numpy as np
import polars as pl
import scipy.sparse as sps

from .._logging import logging
from .._utils import ArgAssert

class RegInference:

    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Regulatory links with {str(self.df.shape[0])} edges"

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        pass


def regulatory_ensure_promoter_included(
        reg: RegInference,
        bin_adata: AnnData,
        gene_adata: AnnData
) -> None:
    toadd = []
    chrs = bin_adata.var['chr'].unique().to_numpy()
    for ch in chrs:
        bin_view = bin_adata[:, bin_adata.var['chr'] == ch]
        gene_view = gene_adata[:, gene_adata.var['chr'] == ch]
        startd = bin_view.var['start'].to_numpy().astype(np.uint64)
        endd = bin_view.var['end'].to_numpy().astype(np.uint64)
        names2 = bin_view.var_names.to_numpy()
        names = gene_view.var_names.to_numpy()
        ix = np.argsort(startd)
        q = gene_view.var['promoter'].to_numpy()
        placeto = np.searchsorted(startd[ix], q, side='right') - 1
        enforcer = (q <= endd[ix][placeto])
        toadd.extend(
            [ch, q[i], q[i], ch, startd[ix][placeto[i]], endd[ix][placeto[i]], 1, 0, 0, names[i],
             names2[ix][placeto[i]]]
            for i in range(gene_view.shape[1]) if enforcer[i]
        )
    newdf = pd.DataFrame(toadd, columns=reg.df.columns)
    newdf = pd.concat((reg.df, newdf))
    reg.df = newdf[~newdf[['name', 'binname']].duplicated()]

def get_regulatory_links(
        reg: RegInference,
        name: str
) -> list[str]:
    """
    Get regulatory links as a list

    Parameters
    ----------
    reg
    name
    """
    df = reg.df
    return df[df['name'] == name]['binname'].to_list()


def get_regulatory_matrix(
        reg: RegInference,
        gene_features: np.ndarray | list[str] | pd.Index,
        bin_features: np.ndarray | list[str] | pd.Index
) -> sps.csr_matrix:
    # TODO: check code
    """
    Get regulatory links as sparse csr matrix.

    Parameters
    ----------
    reg
    gene_features
        Gene features to include
    bin_features
        Bin features to include

    Returns
    -------

    """
    gene_features = np.array(gene_features)
    bin_features = np.array(bin_features)
    ArgAssert(
        (gene_features.shape[0] > 1) and (bin_features.shape[0] > 1),
        "Gene and bin features should be numpy arrays"
    )

    df = pl.DataFrame(reg.df)

    bnnames = np.intersect1d(
        np.array(df.unique('binname')['binname'].to_list()),
        bin_features
    )

    _df = df.filter(pl.col('binname').is_in(bnnames))[['name', 'binname']]

    il = []
    for i in gene_features:
        il.append(
            _df.filter(pl.col('name') == i).to_series(1).to_list()  # .top_k(4, by=pl.col('score'))
        )

    _ix = np.argsort(bin_features)

    coords = []
    for i, _l in enumerate(il):
        _l = _ix[np.searchsorted(bin_features, _l, sorter=_ix)]
        coords.append(np.c_[np.full(_l.shape, fill_value=i), _l])
    coords = np.concat(coords)

    return sps.csr_matrix(
        (np.ones((coords.shape[0],), dtype=np.bool_), (coords[:, 0], coords[:, 1])),
        shape=(gene_features.shape[0], bin_features.shape[0])
    )