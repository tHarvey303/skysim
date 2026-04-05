"""Sersic profile rendering.

Generates 2-D surface brightness stamps for galaxies using the Sersic
profile with JAX. Uses the incomplete gamma function for the exact
Sersic profile (no numerical integration needed).

Key design:
- Precompute b_n for discrete Sersic indices
- Render each galaxy as a small stamp, then paste into the full image
- Sub-seeing galaxies rendered as point sources (LOD optimisation)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc


def sersic_bn(n: jnp.ndarray) -> jnp.ndarray:
    """Approximation for b_n such that gamma(2n, b_n) = 0.5 * Gamma(2n).

    Uses the Ciotti & Bertin (1999) approximation, accurate to <0.1%
    for n > 0.36.
    """
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)


def sersic_profile(
    r: jnp.ndarray,
    r_e: float,
    n: float,
    total_flux: float = 1.0,
) -> jnp.ndarray:
    """1-D Sersic surface brightness profile I(r).

    Parameters
    ----------
    r : array
        Radial distance from center (same units as r_e).
    r_e : float
        Half-light (effective) radius.
    n : float
        Sersic index.
    total_flux : float
        Total integrated flux.

    Returns
    -------
    I : array
        Surface brightness at each r.
    """
    bn = sersic_bn(n)
    # Central surface brightness from total flux
    # F_total = 2*pi*n * r_e^2 * exp(b_n) / b_n^(2n) * gamma(2n)
    from jax.scipy.special import gammaln
    log_norm = (
        jnp.log(total_flux)
        - jnp.log(2.0 * jnp.pi * n)
        - 2.0 * jnp.log(r_e)
        - bn
        + 2.0 * n * jnp.log(bn)
        - gammaln(2.0 * n)
    )
    I_e_norm = jnp.exp(log_norm)
    return I_e_norm * jnp.exp(-bn * ((r / r_e) ** (1.0 / n) - 1.0))


def make_sersic_stamp(
    n: float,
    r_e_pix: float,
    q: float,
    pa: float,
    total_flux: float,
    stamp_size: int,
) -> jnp.ndarray:
    """Render a 2-D Sersic profile stamp.

    Parameters
    ----------
    n : float
        Sersic index.
    r_e_pix : float
        Half-light radius in pixels.
    q : float
        Axis ratio (b/a), in [0, 1].
    pa : float
        Position angle in radians (N through E).
    total_flux : float
        Total flux in the stamp.
    stamp_size : int
        Side length of the square stamp in pixels.

    Returns
    -------
    stamp : array (stamp_size, stamp_size)
        2-D surface brightness image.
    """
    half = stamp_size / 2.0
    y, x = jnp.mgrid[:stamp_size, :stamp_size]
    x = x - half + 0.5
    y = y - half + 0.5

    # Rotate to align with major axis
    cos_pa = jnp.cos(pa)
    sin_pa = jnp.sin(pa)
    x_rot = x * cos_pa + y * sin_pa
    y_rot = -x * sin_pa + y * cos_pa

    # Elliptical radius
    r = jnp.sqrt(x_rot**2 + (y_rot / q) ** 2)
    r = jnp.maximum(r, 0.01)  # avoid division by zero

    stamp = sersic_profile(r, r_e_pix, n, total_flux)
    return stamp


def stamp_size_for_galaxy(
    r_e_pix: float,
    n: float,
    min_size: int = 5,
    max_size: int = 1024,
) -> int:
    """Choose stamp size based on galaxy size and profile shape.

    Higher Sersic n needs more pixels to capture the extended wings.
    Uses 8 + 3*(n-2) effective radii to avoid visible cutoff edges.
    """
    n_re = 8.0 + 3.0 * jnp.clip(n - 2.0, 0.0, 6.0)
    size = int(2.0 * n_re * r_e_pix + 1)
    # Force odd
    size = size + 1 - (size % 2)
    return max(min_size, min(size, max_size))


def is_point_source(
    r_e_pix: jnp.ndarray,
    psf_fwhm_pix: float,
    threshold: float = 0.5,
) -> jnp.ndarray:
    """Determine which galaxies are effectively unresolved.

    Galaxies with R_e < threshold * PSF_FWHM are treated as point sources.
    """
    return r_e_pix < threshold * psf_fwhm_pix


def add_stamp_to_image(
    image: jnp.ndarray,
    stamp: jnp.ndarray,
    cx: int,
    cy: int,
) -> jnp.ndarray:
    """Add a stamp centered at (cx, cy) to the image with boundary clipping."""
    sh, sw = stamp.shape
    hh, hw = sh // 2, sw // 2
    ih, iw = image.shape

    # Stamp slice
    s_y0 = max(0, hh - cy)
    s_x0 = max(0, hw - cx)
    s_y1 = min(sh, ih - cy + hh)
    s_x1 = min(sw, iw - cx + hw)

    # Image slice
    i_y0 = max(0, cy - hh)
    i_x0 = max(0, cx - hw)
    i_y1 = i_y0 + (s_y1 - s_y0)
    i_x1 = i_x0 + (s_x1 - s_x0)

    if i_y1 <= i_y0 or i_x1 <= i_x0:
        return image

    patch = stamp[s_y0:s_y1, s_x0:s_x1]
    return image.at[i_y0:i_y1, i_x0:i_x1].add(patch)


# ---------------------------------------------------------------------------
# Batched stamp rendering (vmap + scatter-add)
# ---------------------------------------------------------------------------

# Stamp-size buckets for batching. Galaxies are rounded up to the nearest
# bucket size so each batch can be vmap'd at a single fixed stamp dimension.
STAMP_BUCKETS = (17, 33, 65, 129, 257, 513, 1025)


def _assign_bucket(stamp_sizes: jnp.ndarray) -> jnp.ndarray:
    """Assign each galaxy to the smallest bucket that fits its stamp."""
    bucket = jnp.full_like(stamp_sizes, STAMP_BUCKETS[-1])
    for b in reversed(STAMP_BUCKETS):
        bucket = jnp.where(stamp_sizes <= b, b, bucket)
    return bucket


def _stamp_sizes_vectorized(
    r_e_pix: jnp.ndarray,
    n: jnp.ndarray,
    min_size: int = 5,
    max_size: int = 1024,
) -> jnp.ndarray:
    """Vectorized stamp size computation (JAX-compatible).

    Fallback when per-galaxy flux is not available. Uses a fixed number
    of effective radii that depends on Sersic index.
    """
    n_re = 8.0 + 3.0 * jnp.clip(n - 2.0, 0.0, 6.0)
    size = (2.0 * n_re * r_e_pix + 1).astype(jnp.int32)
    # Force odd
    size = size + 1 - (size % 2)
    return jnp.clip(size, min_size, max_size)


def _stamp_sizes_from_sb(
    r_e_pix: jnp.ndarray,
    n: jnp.ndarray,
    flux: jnp.ndarray,
    sb_threshold: float,
    min_size: int = 5,
    max_size: int = 1024,
) -> jnp.ndarray:
    """Compute stamp sizes from a surface-brightness threshold.

    Solves for the radius where the Sersic profile drops below
    *sb_threshold* (electrons per pixel), so bright galaxies
    automatically get larger stamps and faint ones get smaller stamps.

    Parameters
    ----------
    r_e_pix : array  – half-light radius in pixels
    n       : array  – Sersic index
    flux    : array  – total flux in electrons
    sb_threshold : float – surface-brightness floor (electrons/pixel)
    """
    from jax.scipy.special import gammaln

    bn = sersic_bn(n)

    # Central surface brightness I_0 (matching sersic_profile normalisation,
    # which does not include axis-ratio q):
    #   I_0 = flux * bn^(2n) / (2*pi*n * r_e^2 * Gamma(2n))
    log_I0 = (
        jnp.log(jnp.maximum(flux, 1e-30))
        + 2.0 * n * jnp.log(bn)
        - jnp.log(2.0 * jnp.pi * n)
        - 2.0 * jnp.log(jnp.maximum(r_e_pix, 0.1))
        - gammaln(2.0 * n)
    )

    # Solve I(r) = I_0 * exp(-bn * (r/re)^(1/n)) = sb_threshold
    #   => r/re = (ln(I_0 / threshold) / bn)^n
    log_ratio = jnp.maximum(log_I0 - jnp.log(sb_threshold), 0.0)
    r_over_re = (log_ratio / bn) ** n
    r_over_re = jnp.clip(r_over_re, 4.0, 200.0)

    size = (2.0 * r_over_re * r_e_pix + 1).astype(jnp.int32)
    size = size + 1 - (size % 2)  # force odd
    return jnp.clip(size, min_size, max_size)


@partial(jax.jit, static_argnums=(5,))
def make_sersic_stamps_batch(
    n: jnp.ndarray,
    r_e_pix: jnp.ndarray,
    q: jnp.ndarray,
    pa: jnp.ndarray,
    flux: jnp.ndarray,
    stamp_size: int,
) -> jnp.ndarray:
    """Render a batch of Sersic stamps at the same fixed size using vmap.

    Parameters
    ----------
    n, r_e_pix, q, pa, flux : arrays of shape (batch,)
    stamp_size : int (static)

    Returns
    -------
    stamps : array (batch, stamp_size, stamp_size)
    """
    half = stamp_size / 2.0
    # Build pixel grid once (shared by all stamps in the batch)
    yy, xx = jnp.mgrid[:stamp_size, :stamp_size]
    xx = (xx - half + 0.5).astype(jnp.float32)
    yy = (yy - half + 0.5).astype(jnp.float32)

    def single(n_i, re_i, q_i, pa_i, f_i):
        cos_pa = jnp.cos(pa_i)
        sin_pa = jnp.sin(pa_i)
        x_rot = xx * cos_pa + yy * sin_pa
        y_rot = -xx * sin_pa + yy * cos_pa
        r = jnp.sqrt(x_rot**2 + (y_rot / q_i) ** 2)
        r = jnp.maximum(r, 0.01)
        return sersic_profile(r, re_i, n_i, f_i)

    return jax.vmap(single)(n, r_e_pix, q, pa, flux)


@partial(jax.jit, static_argnums=(2,))
def add_stamps_scatter(
    image: jnp.ndarray,
    stamps: jnp.ndarray,
    stamp_size: int,
    cx: jnp.ndarray,
    cy: jnp.ndarray,
) -> jnp.ndarray:
    """Add stamps to image using fully vectorized scatter-add.

    Parameters
    ----------
    image : (H, W)
    stamps : (batch, stamp_size, stamp_size)
    stamp_size : int (static)
    cx, cy : (batch,) pixel centers
    """
    h, w = image.shape
    half = stamp_size // 2

    # Offsets within a stamp
    dy, dx = jnp.mgrid[:stamp_size, :stamp_size]
    dy = dy.ravel() - half  # (ss*ss,)
    dx = dx.ravel() - half

    # Global pixel coords for every stamp pixel: (batch, ss*ss)
    gy = cy[:, None] + dy[None, :]
    gx = cx[:, None] + dx[None, :]

    # Flat index into image
    flat_idx = gy * w + gx

    # Validity mask (boundary check)
    valid = (gy >= 0) & (gy < h) & (gx >= 0) & (gx < w)

    flat_idx = jnp.where(valid, flat_idx, 0).astype(jnp.int32)
    values = stamps.reshape(stamps.shape[0], -1)
    values = jnp.where(valid, values, 0.0)

    # Single scatter-add over all stamps
    image_flat = image.ravel()
    image_flat = image_flat.at[flat_idx.ravel()].add(values.ravel())
    return image_flat.reshape(h, w)
