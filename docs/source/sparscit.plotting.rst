Plotting (``sparscit.pl``)
============================

Visualization functions for embeddings, statistics, and comparisons.
All plotting functions return a ``matplotlib.figure.Figure`` when ``show=False``
is passed, allowing further customization.

Embeddings
----------

- :func:`sparscit.plotting._embeddings.embedding2d` — 2D scatter plot of an embedding

Statistics
----------

- :func:`sparscit.plotting._statistics.explained_variance_ratio` — Plot explained variance ratio
- :func:`sparscit.plotting._statistics.depth_corr` — Plot depth-correlation
- :func:`sparscit.plotting._statistics.volcano_plot` — Volcano plot for differential testing

Distributions
-------------

- :func:`sparscit.plotting._hist.counts_histogram` — Histogram of cell counts
- :func:`sparscit.plotting._hist.layer_config_histogram` — Histogram of layer configuration values

Comparisons
-----------

- :func:`sparscit.plotting._compare.compare_categorical` — Compare categorical annotations
- :func:`sparscit.plotting._compare.top_markers` — Plot top marker features
- :func:`sparscit.plotting._compare.frequency_pie` — Pie chart of category frequencies

Heatmaps & Other
----------------

- :func:`sparscit.plotting._heatmap.heatmap` — Heatmap or half-circle plot
- :func:`sparscit.plotting._metabin.metabin` — Plot metadata bin summaries
- :func:`sparscit.plotting._go.goea` — Plot GO enrichment results
- :func:`sparscit.plotting._go.goea_tree` — Plot GO enrichment as a tree
- :func:`sparscit.plotting._inference.regulatory_links` — Visualize regulatory links

.. toctree::
   :maxdepth: 4

   sparscit.plotting.int

.. automodule:: sparscit.plotting
   :members:
   :undoc-members:
   :show-inheritance: