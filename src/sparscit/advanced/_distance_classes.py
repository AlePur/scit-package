from typing import Literal
import scipy.sparse as sps
import numpy as np
import polars as pl
from anndata import AnnData
from typing import Any, Callable

from .._utils._3dsparse import dok_3d_array
from .._utils import ArgAssert
from .._logging import logging
from ..tools._layers import LayerConfig, _format_var_names
from ._dbscan import SubgraphDBSCAN
from ._denoise import denoise_inner

_dist_method = Literal["none", "shortest", "all", "shortest_masked_left", 'shortest_masked_right']

class MultiLayerActivity:

    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"MultiLayerActivity; layers={self.shape[2]}"

    def __init__(self, d: dict, obsm_key: str, shape: tuple):
        self.X = d
        self.future_key = obsm_key
        self.shape = shape

    def to_sps(self, f: Callable | None = None):
        sums = np.array([v.sum() for v in self.X.values()])
        d = []
        ci = []
        ri = []
        j = 0
        from .._logging import PBar
        with PBar.tqdm(total=len(self.X.keys())) as tqdm:
            for i in self.X.keys():
                data = self.X[i]
                if f is not None:
                    data = sps.csr_array(f(sums, j, self.X[i]))
                    data.eliminate_zeros()
                data = data.tocoo()
                d.append(data.data)
                ci.append(data.coords[0])
                ri.append(np.repeat(i, data.data.shape[0]))
                j += 1
                tqdm.n = j
                tqdm.refresh()
        return sps.csr_matrix(
            (np.array([v for row in d for v in row]), (np.array([v for row in ri for v in row]), np.array([v for row in ci for v in row]))),
            shape=(self.shape[0], self.shape[1])
        )

    def collapse(self, threshold: float, auto_range: tuple[float, float]):
        def f(sums, j, X):
            if threshold is not None:
                th = threshold
            else:
                th = (sums[j] / sums.max()) * (auto_range[1] - auto_range[0]) + auto_range[0]
            return (X > th).sum(axis=1)

        return self.to_sps(f)

class FindActive:
    def __init__(
            self,
            adata: AnnData,
            layer_config: LayerConfig,
            datashape: tuple[int, int],
            save_prefix: str | None = None,
            parallel: bool = False,
            *,
            override_data: np.ndarray | None = None
    ) -> None:
        self.adata = adata
        self.layer_config = layer_config
        self.var_mask = layer_config.cached_mask
        if override_data is not None:
            self.Z = override_data
        else:
            self.Z = layer_config.transform(adata, use_cached_mask=True)
        self.genes_to_scan = np.arange(layer_config.get_shape(adata)[1])[self.var_mask]
        self.data = {}
        self.datashape = datashape
        self.parallel = parallel
        self.prefix = "active_cluster_"
        if save_prefix is not None:
            self.prefix = save_prefix

    def run(self, runinternal: Callable) -> MultiLayerActivity:
        from .._logging import PBar

        if self.parallel:
            from multiprocessing import Pool

            pbar = PBar.tqdm(total=self.var_mask.sum())
            pool = Pool(8)

            def _update(*a):
                pbar.update()

            for i in range(pbar.total):
                pool.apply_async(runinternal, args=(self, i), callback=_update)
            pool.close()
            pool.join()
        else:
            with (PBar.tqdm(total=self.var_mask.sum()) as pbar):
                pbar.n = 0

                for gene_index in range(self.var_mask.sum()):
                    pbar.refresh()
                    pbar.n += 1

                    runinternal(
                        self,
                        gene_index
                    )

                pbar.refresh()

        datalen = len(self.data.keys())
        logging.info(f'Recording results for {datalen} genes...')
        if datalen == 0:
            raise ValueError('No genes were found to have active clusters so no results were recorded')

        # make matrix
        pdata = []
        _shape = (self.layer_config.get_shape(self.adata)[1], *self.datashape)

        keys = f"{self.prefix}{self.layer_config.layer}"
        return MultiLayerActivity(self.data, keys, _shape)
        #self.adata.obsm[keys]
        #logging.info(f'Added matrices {keys} to .obsm')