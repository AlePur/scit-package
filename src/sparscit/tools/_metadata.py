from anndata import AnnData
import numpy as np
import re

from .._utils import ArgAssert
from .._logging import logging

def add_metadata(
        adata: AnnData,
        counts_to_add: list[str],
        var_loc_metadata: bool = True
) -> None:
    """
    Add total counts metadata and bin location metadata to an AnnData object.

    For each layer key in ``counts_to_add``, writes per-observation and per-variable
    total counts into ``adata.obs`` and ``adata.var``. If ``var_loc_metadata`` is True,
    parses ``adata.var.index`` (expected format ``chr:start-end``) into separate columns.

    Parameters
    ----------
    adata
        Annotated data matrix
    counts_to_add
        List of layer/obsm keys whose total counts should be added as metadata
    var_loc_metadata
        Whether to parse and add bin location metadata from ``adata.var.index``

    Returns
    -------
    None
        Modifies ``adata`` in place
    """
    def _add_c(k: str, view, skip_v: bool = False):
        adata.obs[f'{k}_total_counts']=(view.sum(axis=1).A1)
        if not skip_v:
            adata.var[f'{k}_total_counts']=(view.sum(axis=0).A1)
        logging.info(f'Added {k} counts metadata')

    if len(counts_to_add) != 0:
        for k in counts_to_add:
            if k in adata.layers.keys():
                _add_c(k, adata.layers[k])
            if k in adata.obsm.keys():
                _add_c(f"obsm_{k}", adata.obsm[k], skip_v=True)

    if var_loc_metadata:
        try:
            sc = np.array([re.split(':|-', a) for a in adata.var.index.tolist()])

            #chr_style = ((len(sc[0][0]) > 3) and (sc[0][0][:3] == 'chr'))
            adata.var['chr'] = sc[:,0]
            adata.var['start'] = sc[:,1].astype(np.int32)
            adata.var['end'] = sc[:,2].astype(np.int32)

        except Exception as e:
            logging.warning("There are problems with the naming of .var indices. This can be caused by anndata objects not generated using this pipeline.\
                          Please fix the issue manually")
            raise e

