"""Image utilities: FFT convolution and subpixel rendering."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def fft_convolve2d(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Convolve a 2-D image with a kernel using FFT.

    The kernel is zero-padded to the image size, and the output
    has the same shape as the input (centered).

    Parameters
    ----------
    image : array (H, W)
        Input image.
    kernel : array (kH, kW)
        Convolution kernel (should sum to 1 for PSF convolution).

    Returns
    -------
    result : array (H, W)
        Convolved image.
    """
    ih, iw = image.shape
    kh, kw = kernel.shape

    # Pad to at least image + kernel - 1 for linear (non-circular) convolution.
    # Use power-of-2 for FFT efficiency.
    fh = _next_power_of_2(ih + kh - 1)
    fw = _next_power_of_2(iw + kw - 1)

    # FFT of both
    image_fft = jnp.fft.rfft2(image, s=(fh, fw))
    kernel_fft = jnp.fft.rfft2(kernel, s=(fh, fw))

    # Multiply and inverse FFT
    result = jnp.fft.irfft2(image_fft * kernel_fft, s=(fh, fw))

    # Crop to original size, accounting for kernel center
    cy, cx = kh // 2, kw // 2
    result = result[cy : cy + ih, cx : cx + iw]
    return result


def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def add_point_source(
    image: jnp.ndarray,
    x: float,
    y: float,
    flux: float,
) -> jnp.ndarray:
    """Add a single point source to the image at subpixel position (x, y).

    Uses bilinear interpolation to distribute flux across 4 pixels.
    """
    ix = jnp.floor(x).astype(int)
    iy = jnp.floor(y).astype(int)
    fx = x - ix
    fy = y - iy

    h, w = image.shape

    w00 = (1 - fx) * (1 - fy) * flux
    w10 = fx * (1 - fy) * flux
    w01 = (1 - fx) * fy * flux
    w11 = fx * fy * flux

    valid = (ix >= 0) & (ix < w - 1) & (iy >= 0) & (iy < h - 1)

    image = jnp.where(valid, image.at[iy, ix].add(w00), image)
    image = jnp.where(valid, image.at[iy, ix + 1].add(w10), image)
    image = jnp.where(valid, image.at[iy + 1, ix].add(w01), image)
    image = jnp.where(valid, image.at[iy + 1, ix + 1].add(w11), image)

    return image


def add_point_sources_batch(
    image: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    flux: jnp.ndarray,
) -> jnp.ndarray:
    """Add many point sources at once using vectorized scatter-add.

    Much faster than calling add_point_source in a loop.
    Uses nearest-pixel placement (no subpixel interpolation) for speed.
    Subpixel accuracy is less important when PSF convolution follows.
    """
    h, w = image.shape

    ix = jnp.floor(x).astype(jnp.int32)
    iy = jnp.floor(y).astype(jnp.int32)

    # Mask out-of-bounds sources
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & (flux > 0)
    ix = jnp.where(valid, ix, 0)
    iy = jnp.where(valid, iy, 0)
    flux_masked = jnp.where(valid, flux, 0.0)

    # Flatten index and scatter-add
    flat_idx = iy * w + ix
    image_flat = image.reshape(-1)
    image_flat = image_flat.at[flat_idx].add(flux_masked)
    return image_flat.reshape(h, w)
