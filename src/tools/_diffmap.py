import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, spdiags, linalg

from anndata import AnnData
from .._utils import ArgAssert, _get_memberships
from .._logging import logging

class Diffmap:
    """
    Scanpy implementation of Hierarchical Diffusion Pseudotime.
    """

    def __init__(
            self,
            connectivities: csr_matrix
    ):
        self.connectivities = connectivities

    def compute_transitions(self, *, density_normalize: bool = True):
        """\
        Compute transition matrix.

        Parameters
        ----------
        density_normalize
            The density rescaling of Coifman and Lafon (2006): Then only the
            geometry of the data matters, not the sampled density.

        Returns
        -------
        Makes attributes `.transitions_sym` and `.transitions` available.
        """
        W = self.connectivities
        # density normalization as of Coifman et al. (2005)
        # ensures that kernel matrix is independent of sampling density
        if density_normalize:
            # q[i] is an estimate for the sampling density at point i
            # it's also the degree of the underlying graph
            q = np.asarray(W.sum(axis=0))
            Q = spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
            K = Q @ W @ Q
        else:
            K = W

        # z[i] is the square root of the row sum of K
        z = np.sqrt(np.asarray(K.sum(axis=0)))
        self.Z = spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
        self.transitions_sym = self.Z @ K @ self.Z

    def compute_eigen(
            self,
            n_comps: int = 15,
            sort_increase: bool = False,
            random_state: int = 0,
    ):
        """\
        Scanpy.
        Compute eigen decomposition of transition matrix.

        Parameters
        ----------
        n_comps
            Number of eigenvalues/vectors to be computed, set `n_comps = 0` if
            you need all eigenvectors.
        random_state
            A numpy random seed

        Returns
        -------
        Writes the following attributes.

        eigen_values : :class:`~numpy.ndarray`
            Eigenvalues of transition matrix.
        eigen_basis : :class:`~numpy.ndarray`
            Matrix of eigenvectors (stored in columns).  `.eigen_basis` is
            projection of data matrix on right eigenvectors, that is, the
            projection on the diffusion components.  these are simply the
            components of the right eigenvectors and can directly be used for
            plotting.
        """
        np.set_printoptions(precision=10)
        if self.transitions_sym is None:
            raise ValueError("Run `.compute_transitions` first.")
        matrix = self.transitions_sym

        n_comps = min(matrix.shape[0] - 1, n_comps)
        # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
        ncv = None
        which = "SM" if sort_increase else "LM"
        # it pays off to increase the stability with a bit more precision
        matrix = matrix.astype(np.float64)

        # Setting the random initial vector
        gen = np.random.default_rng(seed=random_state)
        v0 = gen.standard_normal(matrix.shape[0])
        evals, evecs = linalg.eigsh(
            matrix, k=n_comps, which=which, ncv=ncv, v0=v0
        )
        evals, evecs = evals.astype(np.float32), evecs.astype(np.float32)

        if not sort_increase:
            evals = evals[::-1]
            evecs = evecs[:, ::-1]

        self.eigen_values = evals
        self.eigen_basis = evecs

def diffmap(
        adata: AnnData,
        *,
        obsp_key: str = 'connectivities',
        n_comps: int = 15,
        random_state: int = 0
) -> np.ndarray:
    """
    Diffusion maps. Modified from scanpy

    Parameters
    ----------
    adata
    graph
    n_comps
        Components number
    random_state

    Returns
    -------
    eigen_values
        Eigenvalues for each component
    adata.uns['X_diffmap']
    """
    
    dpt = Diffmap(adata.obsp[obsp_key])
    dpt.compute_transitions()
    dpt.compute_eigen(n_comps=n_comps, random_state=random_state)
    adata.obsm["X_diffmap"] = dpt.eigen_basis
    adata.uns["X_diffmap_evals"] = dpt.eigen_values
    return dpt.eigen_values


def diffusion_pseudotime(
        adata: AnnData,
        root_cell: int
) -> None:
    """\
    Scanpy python package implementation of diffusion pseudotime.
    Infer progression of cells through geodesic distance along the graph
    :cite:p:`Haghverdi2016,Wolf2019`.

    Parameters
    ----------
    adata
    root_cell
        Root cell index

    Returns
    -------
    `adata.obs['dpt_pseudotime']`
        DPT distance with respect to the root cell
    """
    ArgAssert('distances' in adata.obsp.keys(), "Distances not in adata.obsp")
    ArgAssert("X_diffmap" in adata.obsm.keys(), "Diffmap not found in adata.obsm")

    eigen_basis = adata.obsm['X_diffmap']
    eigen_values = adata.uns["X_diffmap_evals"]

    i = root_cell
    pseudotime = sum(
        (
                eigen_values[j] / (1 - eigen_values[j])
                * (eigen_basis[i, j] - eigen_basis[:, j])
        )
        ** 2
        # account for float32 precision
        for j in range(0, eigen_values.size)
        if eigen_values[j] < 0.9994
    )

    pseudotime += sum(
        (eigen_basis[i, k] - eigen_basis[:, k]) ** 2
        for k in range(0, eigen_values.size)
        if eigen_values[k] >= 0.9994
    )
    pseudotime = np.sqrt(pseudotime)
    pseudotime /= np.max(pseudotime[pseudotime < np.inf])

    adata.obs["dpt_pseudotime"] = pseudotime


def normalize_time_per_cluster(
        adata: AnnData,
        time_key: str,
        group_key: str
) -> np.ndarray:
    """
    Normalize time per-cluster. Returns normalized time.

    Parameters
    ----------
    adata
    time_key
    group_key

    Returns
    -------
    time
        Updated time
    """
    cats, cat_names, uq_cats = _get_memberships(adata, group_key)
    times = adata.obs[time_key].to_numpy().copy()
    minmax = (times.min(), times.max())
    for uq in uq_cats:
        times[cats == uq] -= times[cats == uq].min()
        nmax = times[cats == uq].max()
        if nmax < 0.00004:
            logging.warn(f'Extremely small value, probably caused by extremely small group. Skipping group {uq}')
            nmax = np.float32(1)
        times[cats == uq] /= nmax
        times[cats == uq] *= minmax[1] - minmax[0]
        times[cats == uq] += minmax[0]
    return times
