Advanced Tools (``sparscit.adv``)
===================================

Landscape & Grid
----------------

- :func:`src.advanced._landscape_config.create_landscape_scaffold` — Create a ``LandscapeScaffold``
- :func:`src.advanced._griddata.make_grid` — Create a grid for landscape visualization
- :func:`src.advanced._griddata.griddata_numerical` — Interpolate numerical data onto a grid
- :func:`src.advanced._griddata.griddata_categorical` — Interpolate categorical data onto a grid
- :func:`src.advanced._griddata.griddata_apply_transformations` — Apply transformations to gridded data
- :func:`src.advanced._grid.same_grid` — Align two grids

Dynamics & HMMs
----------------

- :func:`src.advanced._dynamics.landscape_dynamics` — Compute landscape dynamics over time
- :func:`src.advanced._dynamics.griddata_correlations` — Compute correlations on gridded data
- :func:`src.advanced._dynamics.dynamics_add_expression_data` — Add expression data to dynamics
- :func:`src.advanced._pomegranate.create_hmm` — Create a Hidden Markov Model
- :func:`src.advanced._pomegranate.create_hmm_system` — Create a multi-cell-type HMM system
- :func:`src.advanced._pomegranate.fit_hmm` — Fit an HMM to data
- :func:`src.advanced._pomegranate.calculate_correlation` — Calculate correlation in HMM context
- :func:`src.advanced._hmm.get_probabilities` — Get state probabilities from an HMM
- :func:`src.advanced._hmm.predict_observations` — Predict observations using an HMM

Community & Enrichment
----------------------

- :func:`src.advanced._community.leiden_matrix` — Leiden clustering returning membership matrix
- :func:`src.advanced._community.random_neighborhood_matrix` — Random neighborhood matrix
- :func:`src.advanced._community.silhouette_score` — Compute silhouette score
- :func:`src.advanced._distances.classify_enrichment` — Classify enrichment patterns
- :func:`src.advanced._distances.neighborhood_activities` — Compute neighborhood activity profiles
- :func:`src.advanced._density.get_assortativity` — Compute graph assortativity

Features & Summaries
--------------------

- :func:`src.advanced._features.binary_active_feature` — Compute binary active feature matrix
- :func:`src.advanced._features.align_features` — Align feature indices between two AnnData objects
- :func:`src.advanced._features.dense_binary_activity_matrix` — Create dense binary activity matrix
- :func:`src.advanced._features.multi_color_copy` — Create multi-color copies of data
- :func:`src.advanced._features.feature_to_bed` — Export features as BED format
- :func:`src.advanced._summary.make_color_summary` — Create a color summary
- :func:`src.advanced._summary.cluster_summaries` — Generate cluster-level summaries
- :func:`src.advanced._summary.cocluster_summaries` — Generate co-cluster summaries
- :func:`src.advanced._entropy.shannon_entropy` — Compute Shannon entropy

Utilities
---------

- :class:`src.advanced._cv2.ImageToolsCV2` — OpenCV-based image processing utilities

.. automodule:: src.advanced
   :members:
   :undoc-members:
   :show-inheritance:
