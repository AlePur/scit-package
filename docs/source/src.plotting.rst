Plotting (``sparscit.pl``)
============================

Visualization functions for embeddings, statistics, and comparisons.
All plotting functions return a ``matplotlib.figure.Figure`` when ``show=False``
is passed, allowing further customization.

Embeddings
----------

- :func:`src.plotting._embeddings.embedding2d` — 2D scatter plot of an embedding

Statistics
----------

- :func:`src.plotting._statistics.explained_variance_ratio` — Plot explained variance ratio
- :func:`src.plotting._statistics.depth_corr` — Plot depth-correlation
- :func:`src.plotting._statistics.volcano_plot` — Volcano plot for differential testing

Distributions
-------------

- :func:`src.plotting._hist.counts_histogram` — Histogram of cell counts
- :func:`src.plotting._hist.layer_config_histogram` — Histogram of layer configuration values

Comparisons
-----------

- :func:`src.plotting._compare.compare_categorical` — Compare categorical annotations
- :func:`src.plotting._compare.top_markers` — Plot top marker features
- :func:`src.plotting._compare.frequency_pie` — Pie chart of category frequencies

Heatmaps & Other
----------------

- :func:`src.plotting._heatmap.heatmap` — Heatmap or half-circle plot
- :func:`src.plotting._metabin.metabin` — Plot metadata bin summaries
- :func:`src.plotting._go.goea` — Plot GO enrichment results
- :func:`src.plotting._go.goea_tree` — Plot GO enrichment as a tree
- :func:`src.plotting._inference.regulatory_links` — Visualize regulatory links

.. toctree::
   :maxdepth: 4

   src.plotting.int

.. automodule:: src.plotting
   :members:
   :undoc-members:
   :show-inheritance:
