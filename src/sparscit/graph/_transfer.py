from anndata import AnnData
import numpy as np
from typing import Any, Literal, Self
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from pandas.api.types import is_numeric_dtype
import pandas as pd

from .._utils._kwargs import _format_kwargs
from .._logging import logging
from .._utils import ArgAssert

_method_type = Literal['average', 'frequent', 'random']
_wtargets = Literal['obs', 'obsm', 'layers', 'uns']

class KnnPredict:
    """
    k-Nearest Neighbor predictor that can transfer data between AnnData objects.

    Stores a trained sklearn KNeighborsClassifier or KNeighborsRegressor
    and uses it to predict values for new data based on a shared embedding.

    Attributes
    ----------
    embedding_key : str
        Key in `.obsm` for the embedding used for prediction
    regression : bool
        Whether the predictor uses regression (True) or classification (False)
    categories : numpy.ndarray or None
        Category labels if classification, None otherwise
    categorical : bool
        Whether the target is categorical
    trained_on : int
        Number of cells the predictor was trained on
    cat_dict : numpy.ndarray or None
        Category dictionary for classification
    """
    def _repr_markdown_(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"""KnnPredict trained on {str(self.trained_on)} cells, 
             regression={self.regression}, embedding_key='{self.embedding_key}'"""

    def __init__(
            self,
            embedding_key: str,
            regression: bool,
            knn_kwargs: dict,
            categorical: bool = False,
            cat_dict: np.ndarray | None = None
    ) -> None:
        self.embedding_key = embedding_key
        self.regression = regression
        self.categories = None
        self.categorical = categorical
        self.trained_on = 0
        self.cat_dict = cat_dict

        if self.regression:
            self.internal = KNeighborsRegressor(**knn_kwargs)
        else:
            self.internal = KNeighborsClassifier(**knn_kwargs)

    def learn(
            self,
            adata: AnnData,
            X: np.ndarray
    ) -> Self:
        self.trained_on = adata.shape[0]
        self.internal.fit(adata.obsm[self.embedding_key], X)
        return self

    def predict(self, *, query_manifold: np.ndarray | None = None, query_adata: AnnData | None = None) -> np.ndarray:
        """
        Predict values for new data using the trained kNN model.

        Parameters
        ----------
        query_manifold
            Pre-computed embedding for the query data. If None, uses ``query_adata.obsm[self.embedding_key]``
        query_adata
            Target AnnData object (used only if ``query_manifold`` is None)

        Returns
        -------
        np.ndarray
            Predicted values (class labels for classification, continuous values for regression)
        """
        logging.info(f'Predicting new data for AnnData based on .obsm[{self.embedding_key}]')
        ArgAssert(self.embedding_key in query_adata.obsm.keys(), f'{self.embedding_key} not found in adata.obsm')
        if query_manifold is None:
            query_manifold = query_adata.obsm[self.embedding_key]
        return self.internal.predict(query_manifold)


def predict(
        adata: AnnData,
        predictor: KnnPredict,
        write_to: _wtargets,
        alias: str
) -> None:
    """
    Predict new values based on trained :class:`KnnPredict`

    Parameters
    ----------
    adata
        Target AnnData
    predictor
        :class:`KnnPredict`
    write_to
        Destination to write to in target AnnData. Possible targets are 'obs', 'uns', 'obsm' and 'layers'
    alias
        Key for newly predicted data
    """

    new = predictor.predict(query_adata=adata)
    if write_to == 'obs':
        new_c = pd.Categorical([predictor.cat_dict[i] for i in new], categories=predictor.cat_dict) if predictor.categorical else new
        adata.obs[alias] = new_c
    elif write_to == 'uns':
        adata.uns[alias] = new
    elif write_to == 'obsm':
        adata.obsm[alias] = new
    elif write_to == 'layers':
        adata.layers[alias] = new
    else:
        raise ValueError(f'Writing to .{write_to} is not supported')

def train_on_obs(
        adata: AnnData,
        embedding_key: str,
        obs_key: str,
        *,
        regression: bool = True,
        n_neighbors: int = 3,
        weights: Literal['uniform', 'distance'] = 'distance'
) -> KnnPredict:
    """
    Train on .obs data using either kNN regression or classification.
    Works for both numerical and categorical data.

    Parameters
    ----------
    adata
        Adata with the obs values to transfer
    embedding_key
        Embedding key of a shared embedding of learning and target adatas
    obs_key
        Obs to transfer
    regression
        Whether training is regression
    n_neighbors
        Number of neighbors to use in training. More leads to more 'smeared' predicted data
    weights
        Weight based on 'distance' or 'uniform'

    Returns
    -------
    :class:`KnnPredict`
        Trained kNN predictor object
    """
    ArgAssert(embedding_key in adata.obsm.keys(), "embedding_key not in adata.obsm")
    ArgAssert(obs_key in adata.obs.keys(), "obs_key not in adata.obs")

    cat_dict = None
    categorical = False
    if adata.obs[obs_key].dtype == "category":
        y = adata.obs[obs_key].cat.codes.values
        if regression:
            logging.warning("Regression is not supported with categorical obs values. Switching to classification...")
            regression = False
        categorical = True
        cat_dict = adata.obs[obs_key].cat.categories
    elif is_numeric_dtype(adata.obs[obs_key]):
        y = adata.obs[obs_key].to_numpy()
    else:
        raise ValueError('The values stored in .obs[obs_key] are neither categorical nor numeric')

    kwargs = _format_kwargs(
        {},
        {
            'n_neighbors': n_neighbors,
            'weights': weights
        }
    )

    return KnnPredict(
        embedding_key,
        regression,
        knn_kwargs=kwargs,
        categorical=categorical,
        cat_dict=cat_dict
    ).learn(
        adata,
        y,
    )


def train_on_matrix(
        adata: AnnData,
        embedding_key: str,
        *,
        layer: str | None = None,
        obsm_key: str | None = None,
        regression: bool = True,
        n_neighbors: int = 3,
        weights: Literal['uniform', 'distance'] = 'distance',
) -> KnnPredict:
    """
    Transfer .layers or .obsm data using either kNN regression or classification.
    One of layer or obsm_key has to be set

    Parameters
    ----------
    adata
        Adata with the layer to transfer
    embedding_key
        Embedding key of a shared embedding of learning and target Anndatas
    layer
        Layer to transfer
    obsm_key
        Embedding to transfer
    regression
        Whether training is regression
    n_neighbors
        Number of neighbors to use in training. More leads to more 'smeared' predicted data
    weights
        Weight based on 'distance' or 'uniform'

    Returns
    -------
    :class:`KnnPredict`
        Trained kNN predictor object that can transfer layer or obsm data
    """
    ArgAssert(embedding_key in adata.obsm.keys(), "embedding_key not in adata.obsm")
    ArgAssert((obsm_key is None) or (layer is None), 'Either layer or obsm_key has to be None')

    use_layer = True
    if obsm_key is not None:
        ArgAssert(obsm_key in adata.obsm.keys(), "obsm_key not in adata.obsm")
        use_layer = False
    elif layer is not None:
        ArgAssert(layer in adata.layers.keys(), "layer not in adata.layers")
    else:
        raise ValueError('Both layer and obsm_key cannot be None')

    kwargs = _format_kwargs(
        {},
        {
            'n_neighbors': n_neighbors,
            'weights': weights
        }
    )

    return KnnPredict(
        embedding_key,
        regression,
        knn_kwargs=kwargs
    ).learn(
        adata,
        adata.layers[layer] if use_layer else adata.obsm[obsm_key],
    )
