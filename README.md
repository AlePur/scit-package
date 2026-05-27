# SparSCit

**Single-cell histone modification sequencing data analysis toolkit**

SparSCit is a Python library for processing, analyzing, and visualizing single-cell histone modification sequencing data (such as NanoC&T, scCUT&Tag, scChIP-seq). It provides a complete pipeline from raw fragment loading through embedding, clustering, landscape visualization, and statistical testing — all built on top of [AnnData](https://anndata.readthedocs.io/).

---

## Installation

```bash
pip install sparscit
```

For development:

```bash
pip install -e .
```

### Requirements

- Python ≥ 3.10
- Key dependencies: `anndata`, `polars`, `scikit-learn`, `scipy`, `numpy`, `jax`, `igraph`, `umap-learn`, `optax`, `goatools`, `matplotlib`, `statsmodels`

See [`requirements.txt`](requirements.txt) for the full list.


---

## Module Overview

SparSCit is organized into six submodules, each accessible via short aliases:

| Submodule | Alias | Purpose |
|-----------|-------|---------|
| `sparscit.load` | `sc.ld` | Data loading (fragments, references, GO terms) |
| `sparscit.tools` | `sc.tl` | Core analysis tools (filtering, statistics, layers) |
| `sparscit.graph` | `sc.gr` | Graph construction, clustering, label transfer |
| `sparscit.embedding` | `sc.em` | Dimensionality reduction and embeddings |
| `sparscit.plotting` | `sc.pl` | Visualization (embeddings, heatmaps, statistics) |
| `sparscit.advanced` | `sc.adv` | Advanced analysis (landscapes, HMMs, dynamics) |

---

## API Reference

### `sparscit.ld` — Loading

Utilities for loading raw data and reference files into AnnData objects.

| Function | Description |
|----------|-------------|
| `fragments(src, genome, ...)` | Load a fragments TSV file into an AnnData object |
| `fragments_to_bulks(src, ...)` | Load fragments and aggregate into bulk profiles |
| `gtf(src, ...)` | Load a GTF reference file, returns a `Reference` object |
| `gaf(src, ...)` | Load a GAF (Gene Association File) for GO annotations |
| `obodag(src, ...)` | Load an OBO DAG for Gene Ontology enrichment |
| `regulatory_links(src, ...)` | Load regulatory link annotations |

### `sparscit.tl` — Tools

Core analysis tools for data processing, statistical testing, and annotation.

#### Data Processing

| Function | Description |
|----------|-------------|
| `filter(adata, layers, ...)` | Filter cells/features by count thresholds (supports percentile cutoffs) |
| `remove_pc(adata, ...)` | Remove principal components from data |
| `transpose_anndata(adata)` | Transpose an AnnData object |
| `stack_adata(adatas, layer_names)` | Stack multiple AnnData objects by adding each as a layer |
| `X_to_layer(adata, layer_name)` | Move `.X` into `.layers` |
| `set_missing_to_zero(adata, features)` | Add missing features as zero-valued columns |
| `copy_to_obs(adata, ...)` | Copy data from layers to `.obs` |
| `make_layer_config(adata, ...)` | Create a `LayerConfig` for reproducible layer transformations |
| `normalize_layers(adata, ...)` | Normalize layers according to a `LayerConfig` |
| `layer_pick_features(adata, ...)` | Select features from a layer |
| `get_layer_feature_means(adata, ...)` | Compute per-feature means from a layer |

#### Statistical Testing

| Function | Description |
|----------|-------------|
| `statistic_test(adata, ...)` | Perform differential testing (t-test, Mann-Whitney U, Wilcoxon, chi-square) |
| `likelihood_test(adata, ...)` | Likelihood ratio test using logistic regression (JAX-accelerated) |
| `likelihood_test_on_adata(adata, ...)` | Convenience wrapper for likelihood testing on AnnData |
| `enriched_in_group(adata, ...)` | Test for enrichment of features in groups |
| `filter_top_markers(markers, ...)` | Filter top marker features from a `Markers` result |

#### Annotation & Metadata

| Function | Description |
|----------|-------------|
| `add_metadata(adata, ...)` | Add metadata columns to `.obs` |
| `add_metabin_metadata(adata, ...)` | Add metadata binned by categorical groups |
| `gene_query(ref, query)` | Query a `Reference` for gene names |
| `region_query(ref, query)` | Query a `Reference` for genomic regions |
| `add_rna_var_metadata(adata, ref)` | Add RNA variable metadata from a reference |
| `symbol2id(ref, genes)` | Convert gene symbols to IDs |
| `get_gene_index(names, feature)` | Get the index of a feature in feature names |

#### Correlation & Comparison

| Function | Description |
|----------|-------------|
| `group_correlation(adata, ...)` | Compute correlations between groups |
| `rep_correlation(adata, ...)` | Compute replication correlation |

#### GO Enrichment

| Function | Description |
|----------|-------------|
| `goea(adata, ...)` | Run Gene Ontology Enrichment Analysis |
| `top_marker_goea(markers, ...)` | Run GOEA on top marker features |

#### Regulation & Inference

| Function | Description |
|----------|-------------|
| `get_regulatory_links(adata, ...)` | Infer regulatory links between features |
| `get_regulatory_matrix(adata, ...)` | Build a regulatory matrix |
| `regulatory_ensure_promoter_included(reg, ...)` | Ensure promoter regions are included in regulatory links |
| `infer_layer(adata, ...)` | Infer a new layer from GO term memberships |
| `go_infer_layer(adata, ...)` | Infer a GO-term-based layer |

#### Diffusion & Pseudotime

| Function | Description |
|----------|-------------|
| `diffmap(adata, ...)` | Compute diffusion map embedding |
| `diffusion_pseudotime(adata, ...)` | Compute diffusion pseudotime |
| `normalize_time_per_cluster(adata, ...)` | Normalize pseudotime per cluster |

#### Other

| Function | Description |
|----------|-------------|
| `kmeans(adata, ...)` | K-means clustering |
| `get_genome_dict(src)` | Get a genome dictionary from a file |
| `summarize_data(adata, ...)` | Summarize data by groups |
| `membership_summary(adata, ...)` | Summarize data by categorical membership |

### `sparscit.gr` — Graph

Graph construction, community detection, and label transfer.

| Function | Description |
|----------|-------------|
| `knn(adata, ...)` | Build a k-nearest-neighbor graph |
| `knn_impute(adata, ...)` | Impute values using k-nearest neighbors |
| `leiden(adata, ...)` | Leiden community detection |
| `community_hierarchy(adata, ...)` | Build a hierarchical community tree from flat clustering |
| `train_on_obs(adata, ...)` | Train a kNN label transfer model on `.obs` values |
| `train_on_matrix(adata, ...)` | Train a kNN label transfer model on matrix data |
| `predict(model, adata, ...)` | Predict labels using a trained kNN model |

### `sparscit.em` — Embedding

Dimensionality reduction and embedding methods.

| Function | Description |
|----------|-------------|
| `pca(adata, ...)` | Principal Component Analysis |
| `spectral(adata, ...)` | Spectral embedding (custom implementation for sparse data) |
| `multiview_spectral(adata, ...)` | Multi-view spectral embedding |
| `umap(adata, ...)` | UMAP embedding |
| `dendrogram(adata, ...)` | Compute dendrogram |
| `landscape(adata, scaffold, ...)` | Create a landscape embedding based on neighborhood clusters |

### `sparscit.pl` — Plotting

Visualization functions for embeddings, statistics, and comparisons.

| Function | Description |
|----------|-------------|
| `embedding2d(adata, key, ...)` | 2D scatter plot of an embedding |
| `heatmap(adata, data, ...)` | Heatmap or half-circle plot |
| `counts_histogram(adata, ...)` | Histogram of cell counts |
| `layer_config_histogram(adata, ...)` | Histogram of layer configuration values |
| `volcano_plot(pval, lfc, ...)` | Volcano plot for differential testing |
| `explained_variance_ratio(adata, ...)` | Plot explained variance ratio |
| `depth_corr(adata, ...)` | Plot depth-correlation |
| `compare_categorical(adata, ...)` | Compare categorical annotations |
| `top_markers(adata, ...)` | Plot top marker features |
| `frequency_pie(adata, ...)` | Pie chart of category frequencies |
| `metabin(adata, ...)` | Plot metadata bin summaries |
| `goea(adata, ...)` | Plot GO enrichment results |
| `goea_tree(adata, ...)` | Plot GO enrichment as a tree |
| `regulatory_links(reg, ...)` | Visualize regulatory links |

### `sparscit.adv` — Advanced

Advanced analysis tools for landscape dynamics, HMMs, and grid-based analysis.

#### Landscape & Grid

| Function | Description |
|----------|-------------|
| `create_landscape_scaffold(adata)` | Create a `LandscapeScaffold` for landscape embedding |
| `make_grid(adata, ...)` | Create a grid for landscape visualization |
| `griddata_numerical(adata, ...)` | Interpolate numerical data onto a grid |
| `griddata_categorical(adata, ...)` | Interpolate categorical data onto a grid |
| `griddata_apply_transformations(adata, ...)` | Apply transformations to gridded data |
| `same_grid(adata, ...)` | Align two grids |

#### Dynamics & HMMs

| Function | Description |
|----------|-------------|
| `landscape_dynamics(adata, ...)` | Compute landscape dynamics over time |
| `griddata_correlations(adata, ...)` | Compute correlations on gridded data |
| `dynamics_add_expression_data(adata, ...)` | Add expression data to dynamics |
| `create_hmm(adata, ...)` | Create a Hidden Markov Model |
| `create_hmm_system(adata, ...)` | Create a multi-cell-type HMM system |
| `fit_hmm(hmm, ...)` | Fit an HMM to data |
| `get_probabilities(hmm, ...)` | Get state probabilities from an HMM |
| `predict_observations(hmm, ...)` | Predict observations using an HMM |

#### Community & Enrichment

| Function | Description |
|----------|-------------|
| `leiden_matrix(adata, ...)` | Run Leiden clustering and return membership matrix |
| `random_neighborhood_matrix(adata, ...)` | Generate random neighborhood matrix |
| `silhouette_score(adata, ...)` | Compute silhouette score |
| `classify_enrichment(adata, ...)` | Classify enrichment patterns |
| `neighborhood_activities(adata, ...)` | Compute neighborhood activity profiles |
| `get_assortativity(adata, ...)` | Compute graph assortativity |

#### Features & Summaries

| Function | Description |
|----------|-------------|
| `binary_active_feature(adata, ...)` | Compute binary active feature matrix |
| `align_features(bin_adata, gene_adata, ...)` | Align feature indices between two AnnData objects |
| `dense_binary_activity_matrix(adata, ...)` | Create a dense binary activity matrix |
| `multi_color_copy(adata, ...)` | Create multi-color copies of data |
| `feature_to_bed(adata, ...)` | Export features as BED format |
| `make_color_summary(adata, ...)` | Create a color summary |
| `cluster_summaries(adata, ...)` | Generate cluster-level summaries |
| `cocluster_summaries(adata, ...)` | Generate co-cluster summaries |
| `shannon_entropy(adata, ...)` | Compute Shannon entropy |

#### Advanced Plotting (`sparscit.adv.pl`)

| Function | Description |
|----------|-------------|
| `landscape2d(adata, ...)` | 2D landscape plot |
| `landscape3d(adata, ...)` | 3D landscape plot |
| `contour2d(adata, ...)` | 2D contour plot |
| `categorical2d(adata, ...)` | 2D categorical plot |
| `slice(adata, ...)` | Slice a landscape |
| `active_subclusters(adata, ...)` | Plot active sub-clusters |
| `community_dendrogram(adata, ...)` | Plot community dendrogram |
| `venn_active_overlap(adata, ...)` | Venn diagram of active feature overlap |
| `time_correlation(adata, ...)` | Plot time correlations |
| `differential_dynamics(adata, ...)` | Plot differential dynamics |
| `summary_overtime(adata, ...)` | Plot summary over time |
| `summary_boxplot(adata, ...)` | Boxplot summary |
| `summary_heatmap(adata, ...)` | Heatmap summary |
| `hidden_states(adata, ...)` | Plot HMM hidden states |
| `hmm_observations(adata, ...)` | Plot HMM observations |
| `hmm_likelihoods(adata, ...)` | Plot HMM likelihoods |
| `transition_matrix(adata, ...)` | Plot HMM transition matrix |
| `custom_cmap(...)` | Create a custom colormap |
| `draw_cbar(...)` | Draw a colorbar |

#### High-Level Helpers (`sparscit.adv.hl`)

| Function | Description |
|----------|-------------|
| `get_cbar(adata, ...)` | Get a colorbar |
| `transform(adata, ...)` | Apply a transformation |
| `plot_with_cbar(adata, ...)` | Plot with an attached colorbar |

---

## Key Classes

### `LayerConfig`

A configuration object that defines how a data layer should be transformed (normalization, log-transform, feature selection, etc.). Created via `sc.tl.make_layer_config()`.

### `Markers`

Result container from `sc.tl.statistic_test()` holding differential testing results as a Polars DataFrame, with methods for filtering top markers.

### `StatisticResult`

Result container from statistical tests, holding test scores and p-values.

### `Graph`

An igraph-backed graph class with support for directed/undirected, weighted edges, and community detection.

### `LandscapeScaffold`

A scaffold for landscape embedding, created via `sc.adv.create_landscape_scaffold()`, specifying cluster memberships and ordering.

### `KnnPredict`

A kNN-based label transfer model, trained via `sc.gr.train_on_obs()` or `sc.gr.train_on_matrix()`.

### `Reference`

A feature reference object loaded from GTF files, used for gene/region queries.

### `RegInference`

A regulatory link inference result, created via `sc.tl.get_regulatory_links()`.

### `Diffmap`

Diffusion map computation class, created via `sc.tl.diffmap()`.

### `HMMWrapper` / `HMMListWrapper`

Hidden Markov Model wrappers for time-series analysis of landscape dynamics.

### `GOEA`

Gene Ontology Enrichment Analysis result container.

### `ImageToolsCV2`

OpenCV-based image processing utilities for landscape visualization.

---

## Global Settings

```python
import sparscit as sc

# Set number of parallel jobs (-1 = all cores)
sc.set_defaults(n_jobs=-1)

# Set default figure size
sc.set_defaults(figsize=(10, 5))

# Enable constrained layout
sc.set_defaults(use_constrained_layout=True)

# Set logging verbosity: 'all', 'no_info', 'error_only'
sc.set_verbosity_level('all')
```

---

## Data Model

SparSCit operates on `anndata.AnnData` objects. Data is stored in:

- **`.X`** — Main data matrix (cells × features)
- **`.layers`** — Additional data layers (e.g., different count matrices, normalized versions)
- **`.obs`** — Cell-level metadata (e.g., cluster assignments, pseudotime)
- **`.var`** — Feature-level metadata (e.g., gene names, genomic coordinates)
- **`.obsm`** — Multi-dimensional annotations (e.g., embeddings like `X_pca`, `X_umap`, `X_landscape`)
- **`.obsp`** — Pairwise annotations (e.g., distance/connectivity matrices)
- **`.uns`** — Unstructured annotations (e.g., community trees, GO results)

---

## Building Documentation

```bash
sphinx-build -M html docs/source/ docs/build/
```

---

## License

See the project repository for license information.