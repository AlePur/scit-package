Advanced Plotting (``sparscit.adv.pl``)
=========================================

Visualization functions for landscape plots, dynamics, HMMs, and summaries.

Landscape Plots
---------------

- :func:`sparscit.advanced.plotting._2d.landscape2d` — 2D landscape plot
- :func:`sparscit.advanced.plotting._2d.contour2d` — 2D contour plot
- :func:`sparscit.advanced.plotting._2d.categorical2d` — 2D categorical plot
- :func:`sparscit.advanced.plotting._3d.landscape3d` — 3D landscape plot
- :func:`sparscit.advanced.plotting._function.slice` — Slice a landscape

Dynamics & HMMs
----------------

- :func:`sparscit.advanced.plotting._dynamics.time_correlation` — Plot time correlations
- :func:`sparscit.advanced.plotting._dynamics.differential_dynamics` — Plot differential dynamics
- :func:`sparscit.advanced.plotting._dynamics.summary_overtime` — Plot summary over time
- :func:`sparscit.advanced.plotting._hmm.hidden_states` — Plot HMM hidden states
- :func:`sparscit.advanced.plotting._hmm.hmm_observations` — Plot HMM observations
- :func:`sparscit.advanced.plotting._hmm.hmm_likelihoods` — Plot HMM likelihoods
- :func:`sparscit.advanced.plotting._hmm.transition_matrix` — Plot HMM transition matrix

Community & Features
--------------------

- :func:`sparscit.advanced.plotting._distances.active_subclusters` — Plot active sub-clusters
- :func:`sparscit.advanced.plotting._dendrogram.community_dendrogram` — Plot community dendrogram
- :func:`sparscit.advanced.plotting._features.venn_active_overlap` — Venn diagram of active feature overlap

Summaries
---------

- :func:`sparscit.advanced.plotting._summary.summary_boxplot` — Boxplot summary
- :func:`sparscit.advanced.plotting._summary.summary_heatmap` — Heatmap summary

Color Utilities
---------------

- :func:`sparscit.advanced.plotting._cmap.custom_cmap` — Create a custom colormap
- :func:`sparscit.advanced.plotting._cmap.draw_cbar` — Draw a colorbar

.. automodule:: sparscit.advanced.plotting
   :members:
   :undoc-members:
   :show-inheritance: