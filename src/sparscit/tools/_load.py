import numpy as np
from typing import Literal

_dict_name = Literal['DM6']


def get_genome_dict(
        name: _dict_name
) -> dict:
    """
    Get a default genome dict by name.

    Parameters
    ----------
    name
        Supported names include 'DM6', 'GRCm39', 'GRCh38'

    Returns
    -------

    """
    if name == 'DM6':
        return {
            "chrX": 23542271,
            "chr2L": 23513712,
            "chr2R": 25286936,
            "chr3L": 28110227,
            "chr3R": 32079331,
            "chr4": 1348131,
            "chrY": 3667352,
            "chrM": 19524
        }
    elif name == 'GRCm39':
        return {
            "chr1": 195154279,
            "chr2": 181755017,
            "chr3": 159745316,
            "chr4": 156860686,
            "chr5": 151758149,
            "chr6": 149588044,
            "chr7": 144995196,
            "chr8": 130127694,
            "chr9": 124359700,
            "chr10": 130530862,
            "chr11": 121973369,
            "chr12": 120092757,
            "chr13": 120883175,
            "chr14": 125139656,
            "chr15": 104073951,
            "chr16": 98008968,
            "chr17": 95294699,
            "chr18": 90720763,
            "chr19": 61420004,
            "chrX": 169476592,
            "chrY": 91455967
        }
    elif name == 'GRCh38':
        return {
            "chr1": 248956422,
            "chr2": 242193529,
            "chr3": 198295559,
            "chr4": 190214555,
            "chr5": 181538259,
            "chr6": 170805979,
            "chr7": 159345973,
            "chr8": 145138636,
            "chr9": 138394717,
            "chr10": 133797422,
            "chr11": 135086622,
            "chr12": 133275309,
            "chr13": 114364328,
            "chr14": 107043718,
            "chr15": 101991189,
            "chr16": 90338345,
            "chr17": 83257441,
            "chr18": 80373285,
            "chr19": 58617616,
            "chr20": 64444167,
            "chr21": 46709983,
            "chr22": 50818468,
            "chrX": 156040895,
            "chrY": 57227415
        }
    else:
        raise ValueError(f"{name} not supported, please make dict manually")