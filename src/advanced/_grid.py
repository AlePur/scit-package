from anndata import AnnData
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap

def _add_to_gd_uns(
    adata: AnnData,
    k: str,
    gd,
):
    if 'griddata' not in adata.uns:
        adata.uns['griddata'] = {}
    adata.uns['griddata'][k] = gd

def _find_bin_indices(
    X: jnp.ndarray,
    grid: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find the bin indices for each point in X.
    
    Args:
        X: Array of shape (n, 2) containing points
        grid: Tuple of (grid_x, grid_y) arrays defining the grid
        
    Returns:
        entries_x: Bin indices along x-axis
        entries_y: Bin indices along y-axis
    """
    entries_x = jnp.searchsorted(grid[0][:, 0][:-1], X[:, 0])
    entries_y = jnp.searchsorted(grid[1][0][:-1], X[:, 1])
    return entries_x, entries_y

def _min_filter(
    z: jnp.ndarray,
    z_m: jnp.ndarray,
    min_cells: int,
    *,
    fill_value: float | int = 0
):
    return jnp.where(z_m > min_cells, z, fill_value)

@partial(jit, static_argnames=['nshape'])
def _map_to_grid_metadata(
    entries_x: jnp.ndarray,
    entries_y: jnp.ndarray,
    nshape: tuple[int, int],
) -> jnp.ndarray:
    """
    Map numerical z values to grid bins using mean aggregation.
    
    Args:
        entries_x: Bin indices along x-axis
        entries_y: Bin indices along y-axis
        z: Numerical values to average in each bin
        nshape: Shape of the output grid
    
    Returns:
        Z: Array of mean z values per bin (bins with no entries are set to 0)
    """
    # Initialize output arrays
    Z_m = jnp.zeros(nshape, dtype=jnp.uint32)
    
    # Use at indexing to accumulate values
    Z_m = Z_m.at[entries_x, entries_y].add(1)
    
    return Z_m

@partial(jit, static_argnames=['nshape'])
def _map_to_grid_numerical(
    entries_x: jnp.ndarray,
    entries_y: jnp.ndarray,
    z: jnp.ndarray,
    nshape: tuple[int, int],
) -> jnp.ndarray:
    """
    Map numerical z values to grid bins using mean aggregation.
    
    Args:
        entries_x: Bin indices along x-axis
        entries_y: Bin indices along y-axis
        z: Numerical values to average in each bin
        nshape: Shape of the output grid
    
    Returns:
        Z: Array of mean z values per bin (bins with no entries are set to 0)
    """
    # Initialize output arrays
    Z_m = jnp.zeros(nshape, dtype=jnp.uint32)
    Z_sum = jnp.zeros(nshape, dtype=z.dtype)
    
    # Use at indexing to accumulate values
    Z_m = Z_m.at[entries_x, entries_y].add(1)
    Z_sum = Z_sum.at[entries_x, entries_y].add(z)
    
    # Compute mean (avoid division by zero)
    Z = jnp.where(Z_m > 0, Z_sum / Z_m, 0)
    
    return Z

@partial(jit, static_argnames=['nshape', 'num_categories'])
def _map_to_grid_categorical(
    entries_x: jnp.ndarray,
    entries_y: jnp.ndarray,
    z: jnp.ndarray,
    nshape: tuple[int, int],
    num_categories: int,
) -> jnp.ndarray:
    """
    Map categorical z values to grid bins using mode (most frequent) aggregation.
    
    Args:
        entries_x: Bin indices along x-axis
        entries_y: Bin indices along y-axis
        z: Integer category values to aggregate in each bin
        nshape: Shape of the output grid
        num_categories: Maximum number of distinct categories (z values should be in range [0, num_categories))
    
    Returns:
        Z: Array of most frequent integer per bin (bins with no entries are set to 0)
    """
    # Flatten grid indices to 1D
    flat_indices = entries_x * nshape[1] + entries_y
    n_bins = nshape[0] * nshape[1]
    
    # Create a 2D histogram: bins x categories
    # For each (bin, category) pair, count occurrences
    bin_cat_indices = flat_indices * num_categories + z
    counts = jnp.bincount(
        bin_cat_indices,
        length=n_bins * num_categories
    )
    
    # Reshape to (n_bins, num_categories)
    counts_2d = counts.reshape(n_bins, num_categories)
    
    # Find the category with max count for each bin
    mode_flat = jnp.argmax(counts_2d, axis=1).astype(jnp.int32)
    
    # Reshape back to grid
    Z = mode_flat.reshape(nshape)
    
    return Z


def _get_image_data(
    X: jnp.ndarray,
    z: jnp.ndarray,
    grid: tuple[jnp.ndarray, jnp.ndarray],
    categorical: bool = False,
    metadata: bool = False
) -> jnp.ndarray:
    """
    JAX implementation of image data binning.
    
    Args:
        X: Array of shape (n, 2) containing points
        z: Array of shape (n, m) containing values
        grid: Tuple of (grid_x, grid_y) arrays defining the grid
    
    Returns:
        Z_sum: Array of summed z values per bin
        Z_m: Array of counts per bin
    """
    # Find bin indices
    entries_x, entries_y = _find_bin_indices(X, grid)
    
    # Create output shape
    nshape = (int(grid[0].shape[0]), int(grid[1].shape[1]))
    
    # Map to grid
    if metadata:
        Z = _map_to_grid_metadata(entries_x, entries_y, nshape)
    else:

        if categorical:
            # It's anyway one layer
            Z = _map_to_grid_categorical(entries_x, entries_y, z, nshape, num_categories=int(z.max()+1))
        else:
            def _map(i: int):
                _z = _map_to_grid_numerical(entries_x, entries_y, z[:, i], nshape)
                return _z
            Z = vmap(_map)(jnp.arange(z.shape[1]))
            
    return Z

def same_grid(
        adatas: list[AnnData]
):
    """
    Copy grid from first anndata to other anndatas
    """
    for i in range(len(adatas)-1):
        adatas[i+1].uns['grid'] = adatas[0].uns['grid'].copy()
        adatas[i+1].uns['summary'] = adatas[0].uns['summary'].copy()