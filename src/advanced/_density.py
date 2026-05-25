
import scipy.stats as sts
import scipy.sparse as sps
import numpy as np
from anndata import AnnData
import pandas as pd
from typing import Self

from sklearn.cluster import DBSCAN

from ..tools._layers import LayerConfig
from .._logging import logging
from .._utils import ArgAssert

class DegreeDistribution:

    def __init__(self) -> None:
        self.dic = {}

    def mean_and_variance(self) -> None:
        xmean = np.array(
            [l[0] * l[1] for l in self.dic.items()]
        ).sum()
        xvar = np.array(
            [((l[0] - xmean)**2) * l[1] for l in self.dic.items()]
        ).sum()
        self.mean = xmean
        self.var = xvar

    @classmethod
    def from_matrix(cls, A: sps.csr_matrix) -> Self:
        binned = np.unique(A.sum(axis=1).A1, return_counts=True)
        d = cls()
        d.dic = dict(list(zip(binned[0], binned[1] / binned[1].sum())))
        d.mean_and_variance()
        return d

    @classmethod
    def from_lower_degree(cls, ld: 'DegreeDistribution') -> Self:
        d = cls()
        _dic = dict([(x, ((x+1) / ld.mean) * ld.get(x+1)) for x in ld.dic.keys()])
        slr = np.array(list(_dic.values())).sum()

        d.dic = dict([(x[0], x[1] / slr) for x in _dic.items()])
        d.mean_and_variance()
        return d

    def get(self, x: int) -> float:
        r = self.dic.get(x)
        return r if r is not None else 0.0

def get_assortativity(
        adata: AnnData,
        layer: LayerConfig,
        *,
        distances_key: str = 'distances'
) -> None:
    """
    Calculate assortativity for all var in Anndata

    Parameters
    ----------
    adata
    layer
        :class:`LayerConfig` with treshold set
    distances_key

    Returns
    -------
    adata.var['assortativity']

    """

    results = np.zeros((adata.shape[1],), dtype=np.float32)
    results_th = np.zeros((adata.shape[1],), dtype=np.float32)

    _X = adata.obsp[distances_key]
    A = (_X + _X.T).astype(np.bool_)
    tX = layer.transform(adata)

    from tqdm.notebook import tqdm
    with tqdm(total=adata.shape[1]) as bar:
        for i in range(adata.shape[1]):
            mask = tX[:, i].sum(axis=1).A1 > layer.get_threshold()
            results_th[i] = mask.sum()
            mA = A[mask][:, mask]
            deg = mA.sum(axis=1).A1
            edges = np.array(mA.tocoo().coords).T

            P = DegreeDistribution.from_matrix(mA)
            q = DegreeDistribution.from_lower_degree(P)

            def e_jk(j: int, k: int) -> float:
                return q.get(j) * q.get(k)

            bar.n += 1
            bar.refresh()

            results[i] = (1 / q.var) * (
                np.array([e_jk(deg[jk[0]],deg[jk[1]]) for jk in edges]).sum() - q.mean**2
            )

    adata.var['passed_th'] = results_th
    adata.var['assortativity'] = results
        