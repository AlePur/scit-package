import numpy as np
from typing import Any
import pandas as pd

from .._utils import ArgAssert
from .._utils._kwargs import _format_kwargs
from ._statistics import Markers
from ._reference import Reference, symbol2id

class GODagWrap:
    """Wrapper around a GO DAG (Gene Ontology Directed Acyclic Graph) object."""

    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "GODagWrap"

    def __init__(self, godag: Any, namespaces: dict[str, str]):
        self.godag = godag
        self.namespaces = namespaces

class GOEA:
    """Gene Ontology Enrichment Analysis result wrapper."""

    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"GOEA result for {self.df.shape[0]} features"

    def __init__(self, res: Any, purge: bool, alpha: float):
        self.res = []

        if purge:
            for r in res:
                if r.p_fdr_bh > alpha:
                    continue
                self.res.append(r)
        else:
            self.res = res
        self.df = self.to_df()

    def to_df(self) -> pd.DataFrame:
        dfraw = []

        for r in self.res:
            dfraw.append(
                [r.NS, r.GO, r.name, r.depth, r.enrichment == 'e', r.pop_count, r.study_count, r.p_fdr_bh,
                 r.p_uncorrected]
            )

        return pd.DataFrame(
            dfraw, columns=[
                'NS', 'GO', 'name', 'depth', 'enriched', 'pop_count', 'study_count', 'p_corrected', 'p_uncorrected'
            ]
        )


def top_marker_goea(
        obodag: GODagWrap,
        ns2assoc: dict,
        markers: Markers,
        ref: Reference | None = None,
        *,
        convert_to_gene_ids: bool = True,
        goea_args_dict: dict = {}
) -> dict[Any, GOEA]:
    """
    Run GO enrichment analysis on top_markers from :class:`Markers`.
    This function provides an option to first convert the marker symbols to gene ids using the
    :class:`Reference` object. This is useful if the gaf (ns2assoc) links ids to GO terms.

    Parameters
    ----------
    obodag
        :class:`GODagWrap`
    ns2assoc
        Associations dict from loading .gaf file
    markers
        :class:`Markers`
    ref
        :class:`Reference`
    convert_to_gene_ids
        Convert gene symbols to ids before running GOEA
    goea_args_dict
        Arguments to pass to scit.tl.goea

    Returns
    -------
    goea_dict
        Dictionary mapping category to :class:`GOEA` result
    """
    population = markers.population

    if convert_to_gene_ids:
        ArgAssert(ref is not None, 'Please provide reference or set convert_to_gene_ids = False')
        population = symbol2id(ref, population)

    ArgAssert(markers.top_markers is not None, "There are no top_markers set in the markers result object")

    kwarg = _format_kwargs(
        goea_args_dict,
        {
            'propagate_counts': True
        }
    )

    res = {}
    for c in markers.top_markers:
        if c[1] is None:
            res[c[0]] = None
            continue

        genes = c[1]['name']
        if convert_to_gene_ids:
            genes = symbol2id(ref, genes)

        res[c[0]] = (
            goea(
                obodag,
                ns2assoc,
                population,
                genes,
                **kwarg
            )
        )
    return res

def goea(
        obodag: GODagWrap,
        ns2assoc: dict,
        population_list: list[str] | np.ndarray,
        queries: list[str] | np.ndarray,
        *,
        alpha: float = 0.05,
        propagate_counts: bool = True,
        include_not_significant: bool = False
) -> GOEA:
    """
    Run GO enrichment analysis.

    Parameters
    ----------
    obodag
        :class:`GODagWrap`
    ns2assoc
        Associations dict from loading .gaf file
    population_list
        Population to test against
    queries
        List of gene ids/symbols to query
    alpha
        Maximum p-value
    propagate_counts
        Whether to propagate counts up in GO term hierarchy
    include_not_significant
        Return non-significant in result

    Returns
    -------
    :class:`GOEA`
    """
    from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
    population_list = list(population_list)
    queries = list(queries)

    ea = GOEA(
        GOEnrichmentStudyNS(
            population_list,  # List of genes
            ns2assoc,  # geneid/GO associations
            obodag.godag,  # Ontologies
            propagate_counts=propagate_counts,
            alpha=alpha,
            methods=['fdr_bh']
        ).run_study(queries),
        not include_not_significant,
        alpha
    )

    return ea