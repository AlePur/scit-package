Advanced Tools (``sparscit.adv``)
===================================

Landscape & Grid
----------------

- :func:`sparscit.advanced._landscape_config.create_landscape_scaffold` — Create a ``LandscapeScaffold``
- :func:`sparscit.advanced._griddata.make_grid` — Create a grid for landscape visualization
- :func:`sparscit.advanced._griddata.griddata_numerical` — Interpolate numerical data onto a grid
- :func:`sparscit.advanced._griddata.griddata_categorical` — Interpolate categorical data onto a grid
- :func:`sparscit.advanced._griddata.griddata_apply_transformations` — Apply transformations to gridded data
- :func:`sparscit.advanced._grid.same_grid` — Align two grids

Dynamics & HMMs
----------------

- :func:`sparscit.advanced._dynamics.landscape_dynamics` — Compute landscape dynamics over time
- :func:`sparscit.advanced._dynamics.griddata_correlations` — Compute correlations on gridded data
- :func:`sparscit.advanced._dynamics.dynamics_add_expression_data` — Add expression data to dynamics
- :func:`sparscit.advanced._pomegranate.create_hmm` — Create a Hidden Markov Model
- :func:`sparscit.advanced._pomegranate.create_hmm_system` — Create a multi-cell-type HMM system
- :func:`sparscit.advanced._pomegranate.fit_hmm` — Fit an HMM to data
- :func:`sparscit.advanced._pomegranate.calculate_correlation` — Calculate correlation in HMM context
- :func:`sparscit.advanced._hmm.get_probabilities` — Get state probabilities from an HMM
- :func:`sparscit.advanced._hmm.predict_observations` — Predict observations using an HMM

Community & Enrichment
----------------------

- :func:`sparscit.advanced._community.leiden_matrix` — Leiden clustering returning membership matrix
- :func:`sparscit.advanced._community.random_neighborhood_matrix` — Random neighborhood matrix
- :func:`sparscit.advanced._community.silhouette_score` — Compute silhouette score
- :func:`sparscit.advanced._distances.classify_enrichment` — Classify enrichment patterns
- :func:`sparscit.advanced._distances.neighborhood_activities` — Compute neighborhood activity profiles
- :func:`sparscit.advanced._density.get_assortativity` — Compute graph assortativity

Features & Summaries
--------------------

- :func:`sparscit.advanced._features.binary_active_feature` — Compute binary active feature matrix
- :func:`sparscit.advanced._features.align_features` — Align feature indices between two AnnData objects
- :func:`sparscit.advanced._features.dense_binary_activity_matrix` — Create dense binary activity matrix
- :func:`sparscit.advanced._features.multi_color_copy` — Create multi-color copies of data
- :func:`sparscit.advanced._features.feature_to_bed` — Export features as BED format
- :func:`sparscit.advanced._summary.make_color_summary` — Create a color summary
- :func:`sparscit.advanced._summary.cluster_summaries` — Generate cluster-level summaries
- :func:`sparscit.advanced._summary.cocluster_summaries` — Generate co-cluster summaries
- :func:`sparscit.advanced._entropy.shannon_entropy` — Compute Shannon entropy

Utilities
---------

- :class:`sparscit.advanced._cv2.ImageToolsCV2` — OpenCV-based image processing utilities

.. automodule:: sparscit.advanced
   :members:
   :undoc-members:
   :show-inheritance: