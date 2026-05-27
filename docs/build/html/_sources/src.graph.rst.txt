Graph Tools (``sparscit.gr``)
==============================

Tools for neighbor-graph construction, community detection, and label transfer.
This submodule builds k-nearest-neighbor graphs, runs Leiden clustering,
constructs community hierarchies, and provides kNN-based label transfer between
datasets.

Neighbor Graphs
---------------

- :func:`src.graph._neighbors.knn` — Build a k-nearest-neighbor graph
- :func:`src.graph._neighbors.knn_impute` — Impute values using k-nearest neighbors

Community Detection
-------------------

- :func:`src.graph._community.leiden` — Leiden community detection
- :func:`src.graph._community.community_hierarchy` — Build hierarchical community tree

Label Transfer
--------------

- :func:`src.graph._transfer.train_on_obs` — Train kNN model on ``.obs`` values
- :func:`src.graph._transfer.train_on_matrix` — Train kNN model on matrix data
- :func:`src.graph._transfer.predict` — Predict labels using a trained kNN model

.. automodule:: src.graph
   :members:
   :undoc-members:
   :show-inheritance:
