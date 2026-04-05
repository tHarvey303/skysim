"""Noise models: Poisson shot noise, read noise, and sky background."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from skysim.config import TelescopeConfig


def sky_background_rate(
    filter_name: str,
    pixel_scale: float,
) -> float:
    """Sky background in electrons/s/pixel.

    Approximate values for common filters. Returns a representative
    sky brightness converted to e-/s/pixel.

    Parameters
    ----------
    filter_name : str
        Filter code (e.g. "JWST/NIRCam.F200W").
    pixel_scale : float
        Pixel scale in arcsec/pixel.

    Returns
    -------
    rate : float
        Background rate in e-/s/pixel.
    """
    # Sky surface brightness in e-/s/arcsec^2 (approximate)
    sky_rates = {
        # JWST (very low background)
        "JWST/NIRCam.F090W": 0.05,
        "JWST/NIRCam.F115W": 0.06,
        "JWST/NIRCam.F150W": 0.08,
        "JWST/NIRCam.F200W": 0.15,
        "JWST/NIRCam.F277W": 0.30,
        "JWST/NIRCam.F356W": 0.50,
        "JWST/NIRCam.F444W": 0.80,
        # HST
        "HST/ACS_WFC.F435W": 0.02,
        "HST/ACS_WFC.F606W": 0.05,
        "HST/ACS_WFC.F814W": 0.08,
        "HST/WFC3_IR.F105W": 0.50,
        "HST/WFC3_IR.F125W": 0.60,
        "HST/WFC3_IR.F160W": 0.80,
    }
    sky_per_arcsec2 = sky_rates.get(filter_name, 0.1)
    return sky_per_arcsec2 * pixel_scale**2


def confusion_noise_rms(
    filter_name: str,
    pixel_scale: float,
    psf_fwhm_arcsec: float,
) -> float:
    """Estimate confusion noise RMS in electrons/s/pixel.

    Uses empirical estimates of the confusion limit from deep surveys.
    The confusion noise scales with the beam solid angle (PSF area).

    Values are approximate floor levels in e-/s/arcsec^2 from unresolved
    sources below ~30th magnitude, based on deep field number counts.

    Parameters
    ----------
    filter_name : str
        Filter code.
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    psf_fwhm_arcsec : float
        PSF FWHM in arcseconds.

    Returns
    -------
    rms : float
        Confusion noise RMS in e-/s/pixel.
    """
    # Empirical confusion surface brightness RMS (e-/s/arcsec^2)
    # at a reference PSF FWHM of 0.1 arcsec. Scales as beam area.
    confusion_ref = {
        "JWST/NIRCam.F090W": 0.002,
        "JWST/NIRCam.F115W": 0.003,
        "JWST/NIRCam.F150W": 0.005,
        "JWST/NIRCam.F200W": 0.008,
        "JWST/NIRCam.F277W": 0.012,
        "JWST/NIRCam.F356W": 0.015,
        "JWST/NIRCam.F444W": 0.020,
        "HST/ACS_WFC.F435W": 0.001,
        "HST/ACS_WFC.F606W": 0.003,
        "HST/ACS_WFC.F814W": 0.005,
        "HST/WFC3_IR.F105W": 0.010,
        "HST/WFC3_IR.F125W": 0.012,
        "HST/WFC3_IR.F160W": 0.015,
    }
    ref_fwhm = 0.1  # arcsec
    base_rate = confusion_ref.get(filter_name, 0.005)
    # Confusion noise scales linearly with beam area (FWHM^2)
    beam_factor = (psf_fwhm_arcsec / ref_fwhm) ** 2
    return base_rate * beam_factor * pixel_scale ** 2


def add_noise(
    key: jax.Array,
    image: jnp.ndarray,
    telescope: TelescopeConfig,
    filter_name: str,
    psf_fwhm_arcsec: float = 0.0,
) -> jnp.ndarray:
    """Add realistic noise to a noiseless image.

    The input image should be in units of electrons (signal * exposure_time).

    Noise sources:
    1. Sky background (Poisson)
    2. Source shot noise (Poisson)
    3. Dark current (Poisson)
    4. Read noise (Gaussian)
    5. Confusion noise from unresolved faint galaxies (Gaussian)

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    image : array (H, W)
        Noiseless image in electrons.
    telescope : TelescopeConfig
    filter_name : str
    psf_fwhm_arcsec : float
        PSF FWHM in arcseconds. Used to estimate confusion noise.
        If 0, confusion noise is skipped.

    Returns
    -------
    noisy : array (H, W)
        Image with noise in electrons.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Sky background
    sky_rate = sky_background_rate(filter_name, telescope.pixel_scale)
    sky_electrons = sky_rate * telescope.exposure_time_s
    sky = sky_electrons * jnp.ones_like(image)

    # Dark current
    dark = telescope.dark_current_e_s * telescope.exposure_time_s * jnp.ones_like(image)

    # Total expected signal (must be non-negative for Poisson)
    total_signal = jnp.maximum(image + sky + dark, 0.0)

    # Poisson noise on total signal
    noisy = jax.random.poisson(k1, total_signal).astype(jnp.float32)

    # Read noise (Gaussian)
    read = telescope.read_noise_e * jax.random.normal(k2, shape=image.shape)
    noisy = noisy + read

    # Confusion noise from unresolved faint galaxies
    if psf_fwhm_arcsec > 0:
        conf_rms_rate = confusion_noise_rms(
            filter_name, telescope.pixel_scale, psf_fwhm_arcsec,
        )
        conf_rms_electrons = conf_rms_rate * telescope.exposure_time_s
        noisy = noisy + conf_rms_electrons * jax.random.normal(k3, shape=image.shape)

    return noisy
