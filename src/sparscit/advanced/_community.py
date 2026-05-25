import numpy as np
from anndata import AnnData
from sklearn.cluster import KMeans
import scipy.sparse as sps
from collections import deque

from .._utils import _get_memberships
from ..graph._community import leiden
from ..graph._neighbors import _neighbor_graph

def leiden_matrix(
        adata: AnnData,
        resolution: float | list[float],
        sample_n: int | None = None,
        *,
        obsp_key: str | None = None,
        beta: float | list[float] = 0.03,
        random_seed: int = 0
) -> None:
    """
    Create a matrix of leiden memberships.

    Parameters
    ----------
    adata
        AnnData
    g
        Neighbor-graph
    resolution
        Resolution parameter for leiden
    sample_n
        Number of independent leiden algorithms to run
    beta
        Beta parameter for leiden
    random_seed
        Initial random seed. Will be incremented by 1 each time new leiden algorithm is run

    Returns
    -------
    adata.obsm['leiden_matrix']
    """

    res = []
    getr = lambda x: resolution
    getb = lambda x: beta
    if sample_n is None:
        if isinstance(resolution, list):
            sample_n = len(resolution)
            getr = lambda x: resolution[x]
        else:
            sample_n = len(beta)
            getb = lambda x: beta[x]
    for i in range(sample_n):
        res.append(leiden(adata, getr(i), random_seed=random_seed, obsp_key=obsp_key, return_raw=True, beta=getb(i)))
        random_seed += 1
    adata.obsm['leiden_matrix'] = np.vstack(res).T


def silhouette_score(
        adata: AnnData,
        obs_key: str,
        embedding_key: str
) -> dict:
    """
    Returns dictionary of silhouette scores per cluster

    Parameters
    ----------
    adata
    obs_key
    embedding_key
    """
    from sklearn.metrics import silhouette_samples
    cats, cat_names, uq_cat = _get_memberships(adata, obs_key)
    sc = silhouette_samples(
        adata.obsm[embedding_key],
        cats
    )
    return {k: np.array(sc[cats == k]) for k in uq_cat}

def random_neighborhood(
        sparse: sps.csr_matrix,
        n_clusters: int,
        random_state: int = 0
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    n = sparse.shape[0]
    memberships = np.full(n, -1, dtype=np.int32)
    
    def partition_nodes(nodes, cluster_id, clusters_remaining):
        if clusters_remaining == 1 or len(nodes) <= 1:
            # Base case: assign all nodes to current cluster
            memberships[nodes] = cluster_id
            return cluster_id + 1
        
        # Calculate target size for this partition
        target_size = len(nodes) // clusters_remaining
        
        # Create subgraph for these nodes
        node_set = set(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Random seed from available nodes
        seed = rng.choice(nodes)
        assigned = np.zeros(len(nodes), dtype=bool)
        assigned[node_to_idx[seed]] = True
        count = 1
        
        # BFS to grow partition
        queue = deque([seed])
        
        while queue and count < target_size:
            current = queue.popleft()
            
            # Get neighbors in the subgraph
            neighbors = sparse[current].indices
            neighbors = [n for n in neighbors if n in node_set and not assigned[node_to_idx[n]]]
            
            # Shuffle for randomness
            rng.shuffle(neighbors)
            
            for neighbor in neighbors:
                if count >= target_size:
                    break
                assigned[node_to_idx[neighbor]] = True
                count += 1
                queue.append(neighbor)
        
        # Split nodes into assigned and unassigned
        partition_1 = nodes[assigned]
        partition_2 = nodes[~assigned]
        
        # Recursively partition each side
        next_cluster_id = partition_nodes(partition_1, cluster_id, 1)
        next_cluster_id = partition_nodes(partition_2, next_cluster_id, clusters_remaining - 1)
        
        return next_cluster_id
    
    # Start with all nodes
    all_nodes = np.arange(n)
    partition_nodes(all_nodes, 0, n_clusters)
    
    return memberships

def random_neighborhood_matrix(
        adata: AnnData,
        n_clusters: int | list[int],
        sample_n: int,
        *,
        obsp_key: str = 'connectivities',
        random_seed: int = 0
) -> None:
    random_seed = int(random_seed)
    res = []
    getn = lambda x: n_clusters
    if isinstance(n_clusters, list):
        getn = lambda x: n_clusters[x]
    for i in range(sample_n):
        res.append(
            random_neighborhood(adata.obsp[obsp_key], n_clusters=getn(i), random_state=random_seed)
        )
        random_seed += 1
    adata.obsm['random_matrix'] = np.vstack(res).T