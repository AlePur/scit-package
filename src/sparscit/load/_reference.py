import polars as pl

from .._utils._kwargs import _format_kwargs
from ..tools._reference import Reference


def gtf(
        src: str,
        feature_name: str = 'gene',
        name_key: str = 'gene_symbol',
        *,
        extra_key: str | None = None,
        polars_csv_args: dict = {}
) -> Reference:
    """
    Load gtf

    Parameters
    ----------
    src
        Path to .gtf
    feature_name
        Feature from gtf (column 3) to use for annotation
    name_key
        Key to use from attributes for feature name
    extra_key
        Extra key to include in :class:`Reference`
    polars_csv_args
        Arguments for polars.read_csv

    Returns
    -------
    :class:`Reference`
    """
    kwargs = _format_kwargs(
        polars_csv_args,
        {
            'separator': '\t',
            'has_header': False,
            'new_columns': ['chr', 'src', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
        }
    )
    csv = pl.read_csv(src, **kwargs).filter(pl.col('feature') == feature_name, )
    # X	FlyBase	CDS	19963955	19964071	.	+	0	gene_id "FBgn0031081"; gene_symbol "Nep3"; transcript_id "FBtr0070000"; transcript_symbol "Nep3-RA";

    chr_style = ((len(csv['chr'][0]) > 3) and (csv['chr'][0][:3] == 'chr'))
    if not chr_style:
        csv = csv.with_columns(
            ("chr" + pl.col("chr")).alias("newchr")
        ).drop("chr").with_columns(pl.col("newchr").alias("chr"))

    csv = csv.with_columns(pl.col("attribute").str.split("; ").alias("atts")) \
        .with_columns(
        pl.col('atts').list.to_struct(
            fields=lambda idx: f"attr_{idx}",
            upper_bound=30
        )
    ).unnest('atts')

    import numpy as np
    cs = np.array(csv[0].columns)
    _cs = cs[np.strings.find(cs, 'attr_') == 0]

    allkeys = [(name_key, 'id')]
    if extra_key is not None:
        allkeys.append((extra_key, 'extra'))
    for (key, keyalias) in allkeys:
        col = None
        for i in _cs:
            if key == csv[0][i][0].split(" ")[0]:
                col = i
                break

        if col is None:
            raise ValueError(f"Could not find key {key} in gtf")

        csv = csv.with_columns(
            pl.col(col).str.extract(r" \"(.*)\"", 1).alias(keyalias)
        )

    flt = ['chr', 'strand', 'start', 'end', 'id']
    if extra_key is not None:
        flt.append('extra')

    return Reference(csv[flt])
