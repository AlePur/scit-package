from anndata import AnnData
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sps

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import Literal

from ...plotting._embeddings import embedding2d
from ..._utils import ArgAssert, _double_lambda
from ..._utils._kwargs import _format_kwargs


def active_subclusters(
        adata: AnnData,
        embedding_key: str,
        obsm_key: str,
        r_index: int,
        *,
        embedding_plot_kwargs: dict = {},
        size: float = 5,
        background_size: float = 2,
        background_alpha: float = 0.5,
        data_override: np.ndarray | None = None,
        make_categorical: bool = True,
        include_arrows: bool = False,
        show: bool = True
) -> Figure | None:
    """
    Plot active subclusters

    Parameters
    ----------
    adata
    embedding_key
    obsm_key
    r_index
    embedding_plot_kwargs
    size
    background_size
    background_alpha
    data_override
    make_categorical
    include_arrows
    show
    """
    kwargs = _format_kwargs(
        embedding_plot_kwargs,
        {
            "black_background": False
        }
    )
    kwargs["show"] = False
    adata.obs['copied_data'] = adata.obsm[obsm_key][:, r_index].todense().A1 if data_override is None else data_override
    includemask = None
    if make_categorical:
        adata.obs['copied_data'] = pd.Categorical(adata.obs['copied_data'])
        includemask = np.array(adata.obs['copied_data'] != 0) * np.array(adata.obs['copied_data'] != "0")

    f = embedding2d(
        adata,
        embedding_key,
        'copied_data',
        size=size,
        include_arrows=include_arrows,
        mask=includemask,
        mask_style_dict={
            'color': 'gray',
            'size': background_size,
            'alpha': background_alpha
        },
        **kwargs
    )

    if show:
        plt.show()
        plt.close(f)
        return None
    return f
