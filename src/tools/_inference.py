from typing import Literal
from anndata import AnnData
import anndata as ad
import numpy as np
import scipy.sparse as sps

from ._layers import LayerConfig
from ._goterm import GODagWrap
from .._logging import logging

def matrix_from_memberships(
        memberships: dict[str, int]
):
    features = list(memberships.keys())
    membership_ids = list(memberships.values())
    
    unique_memberships = np.unique(membership_ids) # integers starting from 0
    n_memberships = len(unique_memberships)
    n_features = len(features)
    
    rows = membership_ids
    cols = list(range(n_features))
    
    R = sps.coo_matrix(
        (np.ones(n_features, dtype=np.bool_), (rows, cols)),
        shape=(n_memberships, n_features)
    ).tocsr()

    #assert matrix_from_memberships(memberships).sum(axis=1).A1 == np.unique(list(memberships.values()), return_counts=True)[1]
    
    return R

def go_infer_layer(
        adata: AnnData,
        gene_population: np.ndarray,
        obodag: GODagWrap,
        ns2assoc: dict,
        include_gos: list[str],
        lc: LayerConfig,
        *,
        return_matrix: bool = False
) -> None:
    """
    """
    namespaces = ['MF', 'CC', 'BP']
    # Make go-term inferred layer

    # Roots from ALL namespaces
    root = []
    for g in obodag.godag.values():
        if g.name in include_gos:
            root.append(g)
            print(g)

    assocs = []
    names = []
    # Gene-go associations
    for namespace in namespaces:
        ids = np.array(list(ns2assoc[namespace].keys()))
        # gos_of_ids
        gos = list(ns2assoc[namespace].values())
        # Include only those which map to gene_population
        mask = np.array([_id in gene_population for _id in ids])
        ids = ids[mask]
        gos_mask = np.zeros((len(gos),), dtype=np.bool_)
        gos_mask[mask] = True

        ix = np.argsort(gene_population)
        id_inserts_to_pop = np.searchsorted(gene_population, ids, sorter=ix)
        id_inserts_to_pop = ix[id_inserts_to_pop]

        # Make assocs
        def edit_mask(term, _mask):
            _mask = _mask + np.array([term.id in g for i, g in enumerate(gos) if gos_mask[i]])
            for c in term.children:
                _mask = edit_mask(c, _mask)
            return _mask

        for r in root:
            mask = edit_mask(
                r,
                np.zeros((ids.shape[0],))
            ).astype(np.bool_)
            assocs.append(id_inserts_to_pop[mask])
            names.append(f"{r.name}_{namespace}")
            logging.info(f"{names[-1]} has {len(assocs[-1])} associated genes")

    import scipy.sparse as sps

    names = np.array(names)
    # assoc is list of lists (outher list len of names)
    from itertools import chain
    rows = np.array(list(chain.from_iterable(assocs)))
    cols = []
    for i in range(len(assocs)):
        cols.extend([i] * len(assocs[i]))

    GO_X = sps.coo_matrix(
        (
            np.ones((len(rows),), dtype=np.bool_),
            (rows, cols)
        ), shape=(lc.get_shape(adata)[1], names.shape[0])
    ).tocsr().T
    
    if return_matrix:
        return GO_X
    else:
        infer_layer(adata, GO_X, lc, names, 'go_term')

_layermode = Literal['sum', 'mean', 'max']


def infer_layer(
        adata: AnnData,
        reg_matrix: sps.csr_matrix,
        layer_config: LayerConfig,
        feature_names: list[str] | np.ndarray,
        new_layer_name: str,
        *,
        use_cached_mask: bool = False,
        mode: _layermode = 'mean'
) -> None:
    """
    Infer layer based on gene-bin regulation. This is useful for creating per-gene acetylation/methylation matrices

    Parameters
    ----------
    adata
    reg_matrix
    layer_config
        Layer to use for inference
    feature_names
        Names of gene-features
    new_layer_name
        Name of new layer
    mode
        Mode of combining bin values

    Returns
    -------
    adata.obsm[new_layer_name]
        New inferred layer
    adata.uns['gene_names_{new_layer_name}']
        Feature names for the new layer
    """
    feature_names = np.array(feature_names)
    sc = reg_matrix.sum(axis=1).A1
    d = layer_config.transform(adata, use_cached_mask=use_cached_mask).astype(np.float32)
    if mode == 'max':
        raise NotImplementedError()
        _X = []
        for i in range(reg_matrix.shape[0]):
            if reg_matrix[i].sum() == 0:
                _X.append(sps.coo_matrix((adata.shape[0],1)))
                continue
            _X.append(d[:, reg_matrix[i].todense().A1].max(axis=1))
        _X = sps.hstack(_X).astype(np.float32)
    else:
        _X = d @ reg_matrix.T
        if mode == 'mean':
            sc[sc == 0] = 1
            _X = _X @ sps.diags(1.0 / sc)
    adata.obsm[new_layer_name] = _X.astype(np.float32)
    adata.uns[f'gene_names_{new_layer_name}'] = feature_names
