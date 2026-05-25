from typing import Any
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps

class custom_array:
    X: Any

    def __getitem__(self, item: int):
        return self.X[item]

class los_3d_array(custom_array):
    def __init__(self, X: list[sps.csr_matrix]):
        self.shape = (len(list), *X[0].shape)
        self.X = X

class dok_3d_array(custom_array):
    def __init__(self, X: dict, shape: tuple[int, ...]):
        self.shape = shape
        self.values = []
        self.keys = []
        for i in range(self.shape[0]):
            p = X.get(i)
            if p is not None:
                self.values.append(p)
                self.keys.append(i)

    def transpose_to_los(self):
        l = np.transpose(axes=(2, 0, 1))

        result = []
        for idx, _l in enumerate(l):
            row_indices = k
            col_indices = np.arange(self.shape[2])

            # Create sparse matrix for each slice
            matrix = sps.csr_matrix(
                (_l.flatten(),
                 (np.repeat(row_indices, len(col_indices)),
                  np.tile(col_indices, len(row_indices)))),
                shape=(self.shape[0], self.shape[2])
            )
            result.append(matrix)
        return los_3d_array(result)