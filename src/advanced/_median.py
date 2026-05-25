import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
import numpy as np
from functools import partial

def median_filter_np_wrapper(image: np.ndarray, size: Tuple[int, int]) -> np.array:
    return np.array(median_filter(jnp.array(image.astype(np.float32)), size))

@partial(jit, static_argnames='size')
def median_filter(image: jnp.ndarray, size: Tuple[int, int]) -> jnp.ndarray:
    """ 
    Applies a 2D median filter to a 4D JAX array along fixed axes (1 and 2). 
    NaN values in the input array are ignored during the median calculation. 
    If a window contains only NaN values, the output for that pixel will be NaN. 

    Parameters: 
    ----------- 
    image : jnp.ndarray 
        The input 4-dimensional image array (e.g., [batch, height, width, channels]). 
    size : Tuple[int, int]
        A tuple of two integers (kernel_s1, kernel_s2) representing the 
        height and width of the median filter kernel. Both dimensions must be odd. 

    Returns: 
    -------- 
    jnp.ndarray 
        The median filtered image of the same shape as the input. 
    """
    kernel_s1, kernel_s2 = size
    
    # Validate inputs
    if image.ndim != 4:
        raise ValueError(f"Input image must be 4-dimensional, but got {image.ndim}D.")
    if len(size) != 2:
        raise ValueError(f"Size must be a tuple of length 2, but got {len(size)}.")
    if not (kernel_s1 % 2 == 1 and kernel_s2 % 2 == 1):
        raise ValueError("Kernel sizes must be odd for symmetric padding.")
    
    pad_1 = kernel_s1 // 2
    pad_2 = kernel_s2 // 2
    batch_size, height, width, channels = image.shape
    
    # Pad the image with NaN values
    padded_image = jnp.pad(
        image.transpose(0, 3, 1, 2), 
        ((0, 0), (0, 0), (pad_1, pad_1), (pad_2, pad_2)), 
        mode='constant', 
        constant_values=jnp.nan
    )
    # NB: Image and padded_image have different order of axes!
    
    def extract_window_and_median(i1: jnp.ndarray, i2: jnp.ndarray) -> jnp.ndarray:
        """Extract a window and compute median for all batch/channel combinations."""
        # Extract the window: shape (batch_size, kernel_s1, kernel_s2, channels)
        window = jax.lax.dynamic_slice(
            padded_image,
            (0, 0, i1, i2),
            (batch_size, channels, kernel_s1, kernel_s2)
        )
        
        # Reshape to (batch_size, channels, kernel_s1 * kernel_s2)
        window_flat = window.reshape(batch_size, channels, -1)
        return jnp.nanmedian(window_flat, axis=-1, keepdims=False)
    
    # First vmap over i2 (width), then over i1 (height)
    extract_row = jax.vmap(extract_window_and_median, (None, 0), 1)
    extract_all = jax.vmap(extract_row, (0, None), 1)
    
    # Create coordinate grids
    i1_coords = jnp.arange(height)
    i2_coords = jnp.arange(width)
    
    result = extract_all(i1_coords, i2_coords)
    
    # out_axis=1 made the result have the right order of axes
    return result