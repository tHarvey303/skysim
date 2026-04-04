"""Point spread function models.

Gaussian, Moffat, and file-based (e.g. WebbPSF) PSF kernels,
normalised to unit sum.

File-based PSFs are loaded from FITS files in a configurable directory.
The naming convention is ``{INSTRUMENT}.{FILTER}.fits``, with ``/``
replaced by ``_``.  For example the filter ``JWST/NIRCam.F444W``
maps to the file ``JWST_NIRCam.F444W.fits``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Default directory for PSF FITS files
PSF_DIR = Path(__file__).resolve().parent.parent / "data" / "psfs"


def gaussian_psf(fwhm_pix: float, size: int = 0) -> jnp.ndarray:
    """2-D Gaussian PSF kernel.

    Parameters
    ----------
    fwhm_pix : float
        Full width at half maximum in pixels.
    size : int
        Kernel side length. If 0, auto-computed as ~10*sigma, forced odd.

    Returns
    -------
    kernel : array (size, size)
        Normalised PSF kernel (sums to 1).
    """
    sigma = fwhm_pix / 2.3548
    if size == 0:
        size = int(10 * sigma + 1)
        size = size + 1 - (size % 2)
    size = max(size, 3)

    half = size / 2.0
    y, x = jnp.mgrid[:size, :size]
    x = x - half + 0.5
    y = y - half + 0.5
    r2 = x**2 + y**2
    kernel = jnp.exp(-0.5 * r2 / sigma**2)
    return kernel / jnp.sum(kernel)


def moffat_psf(
    fwhm_pix: float,
    beta: float = 4.765,
    size: int = 0,
) -> jnp.ndarray:
    """2-D Moffat PSF kernel.

    I(r) = (beta - 1) / (pi * alpha^2) * [1 + (r/alpha)^2]^(-beta)

    Parameters
    ----------
    fwhm_pix : float
        Full width at half maximum in pixels.
    beta : float
        Moffat power-law index (default 4.765 = good fit to atmospheric PSF).
    size : int
        Kernel side length. If 0, auto-computed.

    Returns
    -------
    kernel : array (size, size)
        Normalised PSF kernel (sums to 1).
    """
    alpha = fwhm_pix / (2.0 * jnp.sqrt(2.0 ** (1.0 / beta) - 1.0))

    if size == 0:
        size = int(12.0 * alpha + 1)
        size = size + 1 - (size % 2)
    size = max(size, 3)

    half = size / 2.0
    y, x = jnp.mgrid[:size, :size]
    x = x - half + 0.5
    y = y - half + 0.5
    r2 = x**2 + y**2
    kernel = (1.0 + r2 / alpha**2) ** (-beta)
    return kernel / jnp.sum(kernel)


def _filter_to_filename(filter_code: str) -> str:
    """Convert a filter code like ``JWST/NIRCam.F444W`` to ``JWST_NIRCam.F444W.fits``."""
    return filter_code.replace("/", "_") + ".fits"


def load_psf_fits(
    filter_code: str,
    pixel_scale: float,
    psf_dir: str | Path | None = None,
    psf_pixel_scale: float | None = None,
) -> jnp.ndarray:
    """Load a PSF kernel from a FITS file and resample to the image pixel scale.

    Parameters
    ----------
    filter_code : str
        Filter name, e.g. ``"JWST/NIRCam.F444W"``.
    pixel_scale : float
        Target image pixel scale in arcsec/pixel.
    psf_dir : str or Path, optional
        Directory containing PSF FITS files.  Defaults to ``skysim/data/psfs/``.
    psf_pixel_scale : float, optional
        Pixel scale of the FITS PSF in arcsec/pixel.  If *None* the code
        tries to read it from the FITS header (``PIXELSCL`` or ``CDELT2``).

    Returns
    -------
    kernel : jnp.ndarray (H, W)
        Normalised PSF kernel at the image pixel scale.

    Raises
    ------
    FileNotFoundError
        If no matching FITS file is found.
    """
    from astropy.io import fits

    directory = Path(psf_dir) if psf_dir else PSF_DIR
    fname = _filter_to_filename(filter_code)
    path = directory / fname

    if not path.exists():
        raise FileNotFoundError(
            f"PSF file not found: {path}\n"
            f"Expected naming: {fname} (filter '/' replaced with '_')"
        )

    with fits.open(path) as hdul:
        # Use first image HDU
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                psf_data = np.array(hdu.data, dtype=np.float32)
                header = hdu.header
                break
        else:
            raise ValueError(f"No 2-D image HDU found in {path}")

    # Determine PSF pixel scale from header if not provided
    if psf_pixel_scale is None:
        if "PIXELSCL" in header:
            psf_pixel_scale = float(header["PIXELSCL"])
        elif "CDELT2" in header:
            # CDELT2 is typically in degrees
            psf_pixel_scale = abs(float(header["CDELT2"])) * 3600.0
        else:
            # Assume same pixel scale as image (no resampling)
            psf_pixel_scale = pixel_scale

    # Resample if pixel scales differ
    kernel = _resample_psf(psf_data, psf_pixel_scale, pixel_scale)

    # Normalise
    kernel = kernel / np.sum(kernel)
    return jnp.array(kernel, dtype=jnp.float32)


def _resample_psf(
    psf: np.ndarray,
    psf_scale: float,
    target_scale: float,
) -> np.ndarray:
    """Resample a PSF array from psf_scale to target_scale (arcsec/pix).

    Uses scipy zoom for accurate flux-conserving resampling.
    """
    ratio = psf_scale / target_scale
    if abs(ratio - 1.0) < 1e-3:
        return psf

    from scipy.ndimage import zoom
    resampled = zoom(psf, ratio, order=3)

    # Force odd size for symmetric kernel
    h, w = resampled.shape
    if h % 2 == 0:
        resampled = resampled[:-1, :]
    if w % 2 == 0:
        resampled = resampled[:, :-1]

    return resampled


def list_available_psfs(psf_dir: str | Path | None = None) -> list[str]:
    """List filter codes for which PSF FITS files exist."""
    directory = Path(psf_dir) if psf_dir else PSF_DIR
    if not directory.exists():
        return []
    result = []
    for p in sorted(directory.glob("*.fits")):
        # Reverse the filename → filter_code mapping
        name = p.stem  # e.g. "JWST_NIRCam.F444W"
        # Find the first underscore that should be a "/"
        parts = name.split("_", 1)
        if len(parts) == 2:
            code = parts[0] + "/" + parts[1]
        else:
            code = name
        result.append(code)
    return result


def get_psf_kernel(
    psf_type: str,
    filter_code: str,
    pixel_scale: float,
    psf_fwhm_pix: float,
    psf_dir: str | Path | None = None,
    psf_pixel_scale: float | None = None,
) -> jnp.ndarray:
    """Unified PSF kernel factory.

    Parameters
    ----------
    psf_type : str
        One of ``"gaussian"``, ``"moffat"``, or ``"file"``.
    filter_code : str
        Active filter code (used for file-based PSF lookup).
    pixel_scale : float
        Image pixel scale in arcsec/pixel.
    psf_fwhm_pix : float
        FWHM in pixels (used for gaussian/moffat).
    psf_dir : str or Path, optional
        Directory for PSF FITS files.
    psf_pixel_scale : float, optional
        Pixel scale of FITS PSFs (read from header if None).

    Returns
    -------
    kernel : jnp.ndarray
    """
    if psf_type == "file":
        return load_psf_fits(
            filter_code, pixel_scale,
            psf_dir=psf_dir,
            psf_pixel_scale=psf_pixel_scale,
        )
    elif psf_type == "moffat":
        return moffat_psf(psf_fwhm_pix)
    else:
        return gaussian_psf(psf_fwhm_pix)
