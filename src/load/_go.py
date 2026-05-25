import numpy as np
import anndata as ad
import polars as pl

from ..tools._goterm import GODagWrap
from .._utils import ArgAssert

def obodag(
        obo_file_path: str,
        namespaces: dict[str, str] = {
            'molecular_function': 'MF',
            'cellular_component': 'CC',
            'biological_process': 'BP'
        }
) -> GODagWrap:
    """
    Load OBODAG file for GO term analysis.

    Parameters
    ----------
    obo_file_path
        Path to .obo file
    namespaces
        Dict mapping namespaces in the OBO file to one of 'MF', 'CC', or 'BP'

    Returns
    -------
    :class:`GODagWrap`
    """
    from goatools.obo_parser import GODag

    ArgAssert(
        set(namespaces.values()).issubset({'MF', 'CC', 'BP'}),
        "The namespaces dict must map keys to these values: 'MF', 'CC', 'BP'"
    )
    return GODagWrap(
        GODag(obo_file_path),
        namespaces
    )

def gaf(
        gaf_file_path: str,
        obodag: GODagWrap,
        *,
        name_column: int = 3
) -> dict:
    """
    Load .gaf file associating genes to GO terms.

    Parameters
    ----------
    gaf_file_path
        Path to .gaf file
    obodag
        :class:`GODagWrap`
    name_column
        Which column to use as gene identifier. Column 3 is gene symbol and 2 is gene ID

    Returns
    -------

    """
    ns2assoc = {'MF': {}, 'CC': {}, 'BP': {}}

    g2g_df = pl.scan_csv(
        gaf_file_path,
        has_header=False,
        separator="\t",
        comment_prefix='!'
    ).select(
        pl.col(f"column_{name_column}").alias('symbol'),
        pl.col(f"column_4").str.split('|').alias('verb'),
        pl.col(f"column_5").alias('go')
    ).collect()

    g2g_df = g2g_df.filter(
        ~pl.col('verb').list.contains('NOT')
    )
    g2g = g2g_df[['symbol', 'go']].group_by('symbol').agg('go').to_dict(as_series=False)

    for i in range(len(g2g['symbol'])):
        gene = g2g['symbol'][i]
        gos = g2g['go'][i]

        for k in ns2assoc.keys():
            ns2assoc[k][gene] = set()

        for term in gos:
            ns2assoc[
                obodag.namespaces[obodag.godag[term].namespace]
            ][gene].add(term)

    return ns2assoc
