import numpy as np
import pandas as pd
from anndata import AnnData
from ..tools._layers import LayerConfig
from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt


_alignment = Literal['bin', 'gene']


def align_features(
        bin_adata: AnnData,
        gene_adata: AnnData,
        bin_features: np.ndarray,
        gene_features: np.ndarray
) -> None:
    """
    Align feature indices in two datas.

    Parameters
    ----------
    bin_adata
    gene_adata
    bin_features
    gene_features

    Returns
    -------
    bin_adata.uns['aligned_features'], gene_adata.uns['aligned_features']
        Dict of keys 'names', 'bin_indices', 'gene_indices'
    """
    bin_features = np.array(list(bin_features))
    gene_features = np.array(list(gene_features))
    features = np.intersect1d(gene_features, bin_features)
    cttag_f = np.array([np.argmax(bin_features == f) for f in features])
    rna_f = np.array([np.argmax(gene_features == f) for f in features])
    fd = {
        'names': features,
        'gene_indices': rna_f,
        'bin_indices': cttag_f
    }
    bin_adata.uns['aligned_features'] = fd.copy()
    gene_adata.uns['aligned_features'] = fd.copy()


def binary_active_feature(
        adata: AnnData,
        layer: LayerConfig,
        threshold: float = 0,
        *,
        alignment_type: _alignment | None = None
) -> None:
    """Return a boolean mask of features that are active in at least one cell.

    A feature is considered active if the binarized layer sum across cells
    exceeds ``threshold``.

    Parameters
    ----------
    adata
        AnnData object
    layer
        LayerConfig specifying which layer to transform
    threshold
        Minimum number of active cells for a feature to be considered active
    alignment_type
        If set, restrict the mask to features aligned under the given type
        (``'bin'`` or ``'gene'``)

    Returns
    -------
    Boolean array indicating which features are active
    """
    _all = (layer.transform(adata, binarize=True, return_view_only=True).sum(axis=0).A1 > threshold)
    if alignment_type is not None:
        _all = _all[adata.uns['aligned_features'][f'{alignment_type}_indices']]
    return _all


def multi_color_copy(
        adata: AnnData,
        layers: list[LayerConfig],
        colors: dict = {
            0: '#fff',
            1: '#485fd8',
            10: 'tab:red',
            11: '#c3b812'
        }
) -> np.ndarray:
    """Combine multiple binary layers into a single categorical observation.

    Each layer contributes a digit in a base-10 encoding; the resulting
    categories are stored in ``adata.obs['copied_data']`` and the colour
    mapping in ``adata.uns['colors_dict']``.

    Parameters
    ----------
    adata
        AnnData object to annotate
    layers
        List of LayerConfigs to combine
    colors
        Mapping from encoded category value to colour string

    Returns
    -------
    Boolean mask of cells with non-zero combined activity
    """
    scalar = 10 ** np.arange(len(layers))
    datas = np.array([l.transform(adata, use_cached_mask=True, binarize=True).todense().A1 for l in layers])
    adata.obs['copied_data'] = pd.Categorical(scalar @ datas.astype(np.uint32))
    mask = (adata.obs['copied_data'] != 0).to_numpy()
    adata.uns['colors_dict'] = colors.copy()
    adata.uns['colors_dict'] = {
        str(k): d for k, d in adata.uns['colors_dict'].items() if
        k in adata.obs['copied_data'].cat.categories
    }
    return mask


def feature_to_bed(
        path: str,
        features: np.ndarray,
        mask: np.ndarray
):
    """Write selected features to a BED file.

    Features are expected in ``chr:start-end`` format and are split on
    ``:`` and ``-`` to produce the three required BED columns.

    Parameters
    ----------
    path
        Output BED file path
    features
        Array of feature strings in ``chr:start-end`` format
    mask
        Boolean mask selecting which features to write
    """
    feats = features[mask]
    feats = np.array([
        [f.split(':')[0], *f.split(':')[1].split('-')] for f in feats
    ])
    df = pd.DataFrame(feats, columns=["chrom", "chromStart", "chromEnd"])

    for v in ['name', "score", "strand", "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]:
        df[v] = "."
    df.to_csv(path, sep='\t', index=False, header=False)

def dense_binary_activity_matrix(
        adata: AnnData,
        layer: LayerConfig,
        mask: np.ndarray
) -> np.ndarray:
    """Extract a dense boolean matrix of activity for selected features.

    Parameters
    ----------
    adata
        AnnData object
    layer
        LayerConfig specifying which layer to transform
    mask
        Boolean feature mask selecting which columns to keep

    Returns
    -------
    Dense boolean array of shape (n_cells, n_selected_features)
    """
    l = layer.transform(adata, binarize=True, return_view_only=True)
    return np.array(l[:, mask].todense()).astype(np.bool_)