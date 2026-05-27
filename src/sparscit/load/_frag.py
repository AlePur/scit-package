from anndata import AnnData
import polars as pl
import numpy as np
import anndata as ad
import scipy.sparse as sps
from typing import Literal, OrderedDict
import os
import shutil

from .._logging import PBar
from .._utils._kwargs import _format_kwargs
from .._logging import logging
from .._utils import _get_memberships

_counting_method = Literal['start', 'end', 'both']
_chr_format = Literal['full', 'short', 'auto']

fragSchema = OrderedDict([('chr', pl.String), ('start', pl.UInt64), ('end', pl.UInt64), ('bc', pl.String), ('readSupport', pl.UInt64)])

# Sort keys and values
def get_sorted_kv(gd: dict) -> tuple[np.ndarray, np.ndarray]:
    _skeys = np.array(list(gd.keys()))
    ix = np.argsort(_skeys)

    _skeys = _skeys[ix]
    _svals = np.array(list(gd.values()))[ix]
    return _skeys, _svals

def _short(
        c: str
) -> bool:
    return not ((len(c) > 3) and (c[:3] == 'chr'))

def get_lazy_df(
        src: str,
        kwargs: dict,
        # parquet_temp: str | None = None
):
    lazy_df = pl.scan_csv(
        src, **_filter_dict(
            kwargs,
            ['separator', 'has_header', 'schema', 'comment_prefix']
        ), low_memory=True
    )
    # lazy_df = lazy_df.select(
    #     #pl.int_range(pl.len(), dtype=pl.UInt32).alias("index"),
    #     pl.col('chr'),
    #     pl.col('bc')
    # )#.collect()

    # if parquet_temp is not None:
    #     lazy_df.with_row_index("index").sink_parquet(parquet_temp)
    #     lazy_df = pl.scan_parquet(parquet_temp)
    return lazy_df

def _filter_dict(
        d: dict,
        k_list: list[str]
) -> dict:
    return {k: v for (k, v) in d.items() if np.array([k in k_list]).any()}


def fragments_to_bulks(
        adata: AnnData,
        obs_key: str,
        src: str,
        out_dir: str,
        whitelisted_chr: list,
        *,
        overwrite_output: bool = False,
        override_names: list | None = None,
        polars_csv_kwargs: dict = {},
        gc_collect: bool = True,
        batch_size: int = 50000,
        # parquet_temp: str | None = None,
        chr_name_format: _chr_format = 'auto'
) -> None:
    """Export per-category bulk fragment files from a fragments file.

    Reads a fragments file (potentially gzipped) and writes one TSV per
    category in ``obs_key``, containing the aggregated fragments for all
    cells belonging to that category.

    Parameters
    ----------
    adata
        AnnData with cell metadata in ``.obs``
    obs_key
        Column in ``adata.obs`` defining group memberships
    src
        Path to the fragments TSV file
    out_dir
        Output directory for bulk TSV files
    whitelisted_chr
        List of chromosome names to include
    overwrite_output
        If ``True``, overwrite existing output directory
    override_names
        Optional list of cell names to use instead of ``adata.obs_names``
    polars_csv_kwargs
        Extra keyword arguments passed to the Polars CSV reader
    gc_collect
        Run garbage collection between batches
    batch_size
        Number of rows per batch when reading the fragments file
    chr_name_format
        Chromosome name format: ``'auto'``, ``'ucsc'``, or ``'ensembl'``
    """
    if src[:-3] == '.gz':
        logging.info('You are loading a .gzipped file. Some streaming operations might be impossible, leading to very high memory usage')
    logging.info(f"Writing to {out_dir}(/)<categorical>.tsv")

    if overwrite_output:
        os.makedirs(out_dir, exist_ok=True)
        with os.scandir(out_dir) as d:
            for e in d:
                if e.is_dir():
                    raise Exception(f"The out_dir contains a subdirectory '{e.name}'. Please clear manually.")

                if e.name[-4:] != '.tsv':
                    raise Exception('The out_dir contains files that are not tsv. Please clear the directory yourself')

        shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=False)

    kwargs = _format_kwargs(
        polars_csv_kwargs,
        {
            'separator': '\t',
            'has_header': False,
            'schema': fragSchema,
            'new_columns': list(fragSchema.keys()),
            'schema_overrides': list(fragSchema.values()),
            'comment_prefix': '#',
            'batch_size': batch_size
        }
    )

    logging.info("Loading lazy...")

    lazy_df = get_lazy_df(src, kwargs)#, parquet_temp)

    # Get memberships
    cats, cat_names, uq_cats = _get_memberships(adata, obs_key)
    logging.info(f"Categoricals: {cat_names}")
    
    names = list(adata.obs_names)
    if override_names is not None:
        names = override_names
    df_cat = pl.DataFrame([np.array(names), cats], {'bc': pl.String, 'cat_code': pl.UInt32})

    # Logic for able to ignore chr

    nlines = lazy_df.select(
            pl.len().alias("nlines")
        ).collect()["nlines"][0]
    logging.info(f"Reading fragments file with {nlines} lines...")
    uq_chr = lazy_df.select(pl.col("chr").unique()).collect().to_series().to_list()
    bcs = np.sort(
        lazy_df.select(pl.col("bc").unique()).collect().to_series().to_numpy()
    )

    logging.info("Done loading")

    logging.info(f"Number of unique barcodes is {bcs.shape[0]}...")

    if chr_name_format == 'auto':
        use_shortkeys = np.array([_short(c) for c in uq_chr]).all() and (
            not np.array([_short(c) for c in whitelisted_chr]).any())
        if use_shortkeys:
            logging.info(
                "Found no chromosomes starting with 'chr' in fragments. Mapping chromosome names with lambda x: "
                "'chr' + x. If you want to avoid this, set chr_name_insert_prefix = False"
            )
    else:
        use_shortkeys = chr_name_format == 'short'

    if use_shortkeys:
        uq_chr = np.array([f"chr{u}" for u in uq_chr])
    else:
        uq_chr = np.array(uq_chr)
    uq_chr = np.sort(uq_chr)

    mask_keep_chr = np.array([c in whitelisted_chr for c in uq_chr])
    ignored_chr = uq_chr[~mask_keep_chr]

    if len(list(ignored_chr)) != 0:
        logging.info(f'Ignoring chromosomes {ignored_chr}')

    # 2L	4646	5306	AGTCAACCACGTTAGT-1	1
    reader = pl.read_csv_batched(src, **{k: v for (k, v) in kwargs.items() if k != 'schema'})

    # estimate batch number
    n_batches = int(nlines / (kwargs['batch_size'])) + 1
    batches = reader.next_batches(1)

    # if parquet_temp is not None:
    #     os.remove(parquet_temp)

    if gc_collect:
        del lazy_df
        import gc; gc.collect()

    with PBar.tqdm(total=n_batches) as pbar:
        while batches:
            df = batches[0]
            df_w_annot = df.join(df_cat, on='bc', how='inner', suffix='_annot')
    
            if df_w_annot.shape[0] == 0:
                logging.warn('There are 0 barcodes in union between fragment bcs & annotated bcs')
            for uq_i in range(len(uq_cats)):
                # ['chr', 'start', 'end', 'bc', 'readSupport']
                with open(f"{out_dir}/{cat_names[uq_i]}.tsv", mode="a") as f:
                    df_w_annot.filter(pl.col('cat_code')==uq_cats[uq_i])[['chr', 'start', 'end', 'bc']].write_csv(f, include_header=False, separator='\t')

            pbar.total = max([pbar.n + 1, pbar.total])
            pbar.n += 1
            pbar.refresh()
            batches = reader.next_batches(1)

        pbar.total = min([pbar.n, pbar.total])
        pbar.refresh()

def fragments(
        src: str,
        genome_dict: dict,
        bin_size: int = 5000,
        *,
        method: _counting_method = 'both',
        polars_csv_kwargs: dict = {},
        ignore_missing_chr: bool = False,
        gc_collect: bool = True,
        chr_name_format: _chr_format = 'auto'
) -> AnnData:
    """
    Load fragments file outputted by Cellranger ATAC pipeline.

    Parameters
    ----------
    src
        Path to fragments file
    genome_dict
        Dict with chromosome names (formatted as "chr<x>") and chromosome sizes
    bin_size
        Bin size in bases
    method
        Method of counting reads
    polars_csv_kwargs
        polars kwargs
    ignore_missing_chr
        Whether to ignore missing chromosomes in genome_dict or raise an error
    chr_name_format
        If this is 'short', insert 'chr' prefix if it is missing in fragments or not.

    Returns
    -------
    :class:`AnnData`
        Reads are placed in `adata.X`. Obs indices are cell barcodes and var indices are `chr:start-end`
    """
    #bin_size = np.array(bin_size, dtype=np.float32)
    genome_dict = genome_dict.copy()  # Will be modified
    use_start = False
    use_end = False
    if method == 'end' or method == 'both':
        use_end = True
    if method == 'start' or method == 'both':
        use_start = True

    kwargs = _format_kwargs(
        polars_csv_kwargs,
        {
            'separator': '\t',
            'has_header': False,
            'schema': fragSchema,
            'new_columns': list(fragSchema.keys()),
            'schema_overrides': list(fragSchema.values()),
            'comment_prefix': '#',
            'batch_size': 50000 # There is a polars bug, this value is now respected
        }
    )

    skeys, svals = get_sorted_kv(
        genome_dict
    )

    lazy_df = pl.scan_csv(
        src, **_filter_dict(
            kwargs,
            ['separator', 'has_header', 'schema', 'comment_prefix']
        )
    )
    lazy_df = lazy_df.select(
        pl.int_range(pl.len(), dtype=pl.UInt64).alias("index"),
        pl.col('chr'),
        pl.col('bc')
    ).collect()

    # Logic for able to ignore chr

    nlines = lazy_df['index'][-1] + 1
    logging.info(f"Reading fragments file with {nlines} lines...")
    uq_chr = lazy_df['chr'].unique().to_list()
    bcs = np.sort(lazy_df['bc'].unique().to_numpy())
    logging.info(f"Number of unique barcodes is {bcs.shape[0]}...")

    if chr_name_format == 'auto':
        use_shortkeys = np.array([_short(c) for c in uq_chr]).all() and (
            not np.array([_short(c) for c in list(genome_dict.keys())]).any())
        if use_shortkeys:
            logging.info(
                "Found no chromosomes starting with 'chr' in fragments. Mapping chromosome names with lambda x: "
                "'chr' + x. If you want to avoid this, set chr_name_insert_prefix = False"
            )
    else:
        use_shortkeys = chr_name_format == 'short'

    if use_shortkeys:
        uq_chr = np.array([f"chr{u}" for u in uq_chr])
    else:
        uq_chr = np.array(uq_chr)
    uq_chr = np.sort(uq_chr)

    mask_keep_chr = np.array([c in skeys for c in uq_chr])
    ignored_chr = uq_chr[~mask_keep_chr]
    ignored_chr_idx = np.arange(uq_chr.shape[0])[~mask_keep_chr]

    if len(list(ignored_chr)) != 0:
        if not ignore_missing_chr:
            raise ValueError(
                f'Not all chromosomes (keys {ignored_chr}) are found in genome_dict. Please add them or set '
                f'ignore_missing_chr = True'
            )
        logging.info(f'Ignoring chromosomes {ignored_chr}')

    lengths_dict = {}
    for uc in uq_chr:
        gd_value = genome_dict.get(uc)
        lengths_dict[uc] = gd_value if gd_value is not None else bin_size
    skeys, svals = get_sorted_kv(
        lengths_dict
    )

    # Make bin ranges
    lens = np.array([
        np.floor(np.array(v, dtype=np.float32) / np.array(bin_size, dtype=np.float32))
        + 1 for v in svals
    ], dtype=np.int32)
    bin_lens = lens.copy()

    # Offset to account for other chromosomes in M index
    lens[-1] = 0
    lens = np.roll(lens, shift=1)
    clens = np.cumsum(lens).astype(np.uint32)

    # 2L	4646	5306	AGTCAACCACGTTAGT-1	1
    reader = pl.read_csv_batched(src, **{k: v for (k, v) in kwargs.items() if k != 'schema'})

    # estimate batch number
    n_batches = int(nlines / (kwargs['batch_size'])) + 1
    batches = reader.next_batches(1)

    # Init
    N = []
    M = []
    if gc_collect:
        del lazy_df
        import gc; gc.collect()

    with PBar.tqdm(total=n_batches) as pbar:
        while batches:
            df = batches[0]

            sub_map = np.searchsorted(bcs, df['bc'].to_numpy())

            df = df.with_columns(
                pl.Series(name="idx", values=sub_map)
            )

            if use_shortkeys:
                df = df.with_columns(
                    ('chr' + pl.col('chr')).alias('chr')
                )

            chr_idx = np.searchsorted(skeys, df['chr'], side='left')
            df = df.with_columns(
                pl.Series(name="chr_idx", values=chr_idx)
            )
            df = df.filter(~pl.col("chr_idx").is_in(ignored_chr_idx), )
            to_check = df['chr_idx'].unique().to_list()

            for chrom in to_check:
                if chrom == skeys.shape[0]:
                    raise ValueError("The fragments file contains chromosomes that are not provided in the genome_dict")

                _df = df.filter(pl.col("chr_idx") == chrom, )

                # Init arrays to be added to M and N
                m_add = []
                n_add = [_df['idx'].to_numpy().astype(np.uint64)]

                if use_start:
                    pos = np.floor(_df['start'].to_numpy() / np.array(bin_size, dtype=np.float32)).astype(np.uint64)
                    m_add = [pos + clens[chrom]]  # list

                if use_end:
                    epos = np.floor(_df['end'].to_numpy() / np.array(bin_size, dtype=np.float32)).astype(np.uint64)
                    _m_add = epos + clens[chrom]

                    if use_start:
                        # Basically add end conditionally if use_start is used (method=double)

                        msk = ((pos - epos) != 0)
                        # these barcodes get added twice:
                        n_add.append(n_add[0][msk])
                        m_add.append(_m_add[msk])
                    else:
                        m_add = [_m_add]

                M.extend(m_add)
                N.extend(n_add)

            pbar.total = max([pbar.n + 1, pbar.total])
            pbar.n += 1
            pbar.refresh()
            batches = reader.next_batches(1)

        pbar.total = min([pbar.n, pbar.total])
        pbar.refresh()

    mxshape = (bcs.shape[0], int(clens[-1] + bin_lens[-1]))

    var_mask = []
    vari = []
    for i, k in enumerate(skeys):
        vari.extend([f"{k}:{j*bin_size}-{(j+1)*bin_size}" for j in range(bin_lens[i])])
        if k not in ignored_chr:
            var_mask.extend([True for j in range(bin_lens[i])])
        else:
            var_mask.extend([False for j in range(bin_lens[i])])

    M = np.concat(M, dtype=np.uint32)
    N = np.concat(N, dtype=np.uint32)
    sps_acc = sps.coo_matrix((np.ones(N.shape, dtype=np.uint32), (N, M)), shape=mxshape, dtype=np.uint32).tocsr()
    data = ad.AnnData(sps_acc)
    data.obs.index = bcs
    data.var.index = vari
    return data[:, np.array(var_mask)]
