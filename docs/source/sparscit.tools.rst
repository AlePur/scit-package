Tools (``sparscit.tl``)
========================

Core analysis tools for data processing, statistical testing, and annotation.
This is the largest submodule, covering filtering, normalization, statistical
tests, GO enrichment, regulatory inference, diffusion maps, and more.

Data Processing
---------------

- :func:`sparscit.tools._filter.filter` — Filter cells/features by count thresholds
- :func:`sparscit.tools._filter.remove_pc` — Remove principal components
- :func:`sparscit.tools._transpose.transpose_anndata` — Transpose an AnnData object
- :func:`sparscit.tools._combine.stack_adata` — Stack multiple AnnData objects as layers
- :func:`sparscit.tools._combine.X_to_layer` — Move ``.X`` into ``.layers``
- :func:`sparscit.tools._combine.set_missing_to_zero` — Add missing features as zero columns
- :func:`sparscit.tools._layers.copy_to_obs` — Copy data from layers to ``.obs``
- :func:`sparscit.tools._layers.make_layer_config` — Create a ``LayerConfig``
- :func:`sparscit.tools._layers.normalize_layers` — Normalize layers
- :func:`sparscit.tools._layers.layer_pick_features` — Select features from a layer
- :func:`sparscit.tools._layers.get_layer_feature_means` — Compute per-feature means

Statistical Testing
-------------------

- :func:`sparscit.tools._statistics.statistic_test` — Differential testing (t-test, Mann-Whitney U, Wilcoxon, chi-square)
- :func:`sparscit.tools._statistics.likelihood_test` — Likelihood ratio test (JAX-accelerated)
- :func:`sparscit.tools._statistics.likelihood_test_on_adata` — Convenience wrapper for likelihood testing
- :func:`sparscit.tools._statistics.enriched_in_group` — Test for feature enrichment in groups
- :func:`sparscit.tools._statistics.filter_top_markers` — Filter top markers from a ``Markers`` result

Annotation & Metadata
---------------------

- :func:`sparscit.tools._metadata.add_metadata` — Add metadata to ``.obs``
- :func:`sparscit.tools._metabin.add_metabin_metadata` — Add binned metadata
- :func:`sparscit.tools._reference.gene_query` — Query a ``Reference`` for gene names
- :func:`sparscit.tools._reference.region_query` — Query a ``Reference`` for genomic regions
- :func:`sparscit.tools._reference.add_rna_var_metadata` — Add RNA variable metadata
- :func:`sparscit.tools._reference.symbol2id` — Convert gene symbols to IDs
- :func:`sparscit.tools._reference.get_gene_index` — Get feature index by name

Correlation & Comparison
-------------------------

- :func:`sparscit.tools._corr.group_correlation` — Compute group correlations
- :func:`sparscit.tools._corr.rep_correlation` — Compute replication correlation

GO Enrichment
-------------

- :func:`sparscit.tools._goterm.goea` — Gene Ontology Enrichment Analysis
- :func:`sparscit.tools._goterm.top_marker_goea` — GOEA on top markers

Regulation & Inference
----------------------

- :func:`sparscit.tools._regulation.get_regulatory_links` — Infer regulatory links
- :func:`sparscit.tools._regulation.get_regulatory_matrix` — Build regulatory matrix
- :func:`sparscit.tools._regulation.regulatory_ensure_promoter_included` — Ensure promoters in regulatory links
- :func:`sparscit.tools._inference.infer_layer` — Infer a new layer from GO memberships
- :func:`sparscit.tools._inference.go_infer_layer` — Infer a GO-term-based layer

Diffusion & Pseudotime
----------------------

- :func:`sparscit.tools._diffmap.diffmap` — Compute diffusion map
- :func:`sparscit.tools._diffmap.diffusion_pseudotime` — Compute diffusion pseudotime
- :func:`sparscit.tools._diffmap.normalize_time_per_cluster` — Normalize pseudotime per cluster

Other
-----

- :func:`sparscit.tools._kmeans.kmeans` — K-means clustering
- :func:`sparscit.tools._load.get_genome_dict` — Get genome dictionary
- :func:`sparscit.tools._metabin.summarize_data` — Summarize data by groups
- :func:`sparscit.tools._metabin.membership_summary` — Summarize data by categorical membership

.. automodule:: sparscit.tools
   :members:
   :undoc-members:
   :show-inheritance: