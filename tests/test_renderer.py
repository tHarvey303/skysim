"""Tests for the image rendering pipeline."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from skysim.models.psf import gaussian_psf, moffat_psf
from skysim.telescope.noise import add_noise, sky_background_rate
from skysim.utils.image import fft_convolve2d
from skysim.config import TelescopeConfig


def test_gaussian_psf_normalised():
    """Gaussian PSF should sum to 1."""
    kernel = gaussian_psf(3.0)
    npt.assert_allclose(float(jnp.sum(kernel)), 1.0, atol=1e-5)


def test_moffat_psf_normalised():
    """Moffat PSF should sum to 1."""
    kernel = moffat_psf(3.0)
    npt.assert_allclose(float(jnp.sum(kernel)), 1.0, atol=1e-5)


def test_gaussian_psf_symmetric():
    kernel = gaussian_psf(5.0, size=21)
    npt.assert_allclose(kernel, kernel[::-1, :], atol=1e-6)
    npt.assert_allclose(kernel, kernel[:, ::-1], atol=1e-6)


def test_moffat_wider_wings():
    """Moffat should have more flux in the wings than Gaussian."""
    g = gaussian_psf(5.0, size=41)
    m = moffat_psf(5.0, size=41)
    # Compare flux outside a central 11x11 box
    mask = jnp.ones((41, 41)).at[15:26, 15:26].set(0.0)
    g_wings = float(jnp.sum(g * mask))
    m_wings = float(jnp.sum(m * mask))
    assert m_wings > g_wings


def test_fft_convolve_delta():
    """Convolving with a delta function should return the kernel."""
    image = jnp.zeros((64, 64))
    image = image.at[32, 32].set(1.0)
    kernel = gaussian_psf(3.0, size=11)
    result = fft_convolve2d(image, kernel)
    # Peak should be near (32, 32)
    peak = jnp.unravel_index(jnp.argmax(result), result.shape)
    assert abs(int(peak[0]) - 32) <= 1
    assert abs(int(peak[1]) - 32) <= 1


def test_fft_convolve_flux_conservation():
    """Convolution should conserve total flux."""
    image = jnp.zeros((64, 64))
    image = image.at[32, 32].set(100.0)
    kernel = gaussian_psf(3.0, size=11)
    result = fft_convolve2d(image, kernel)
    npt.assert_allclose(float(jnp.sum(result)), 100.0, rtol=0.01)


def test_noise_adds_variance():
    """Noisy image should have more variance than noiseless."""
    key = jax.random.PRNGKey(0)
    image = 100.0 * jnp.ones((64, 64))
    tel = TelescopeConfig(exposure_time_s=100.0, read_noise_e=4.0)
    noisy = add_noise(key, image, tel, "JWST/NIRCam.F200W")
    assert float(jnp.std(noisy)) > float(jnp.std(image))


def test_noise_mean_close_to_signal():
    """Mean of noisy image should be close to signal + background."""
    key = jax.random.PRNGKey(42)
    signal = 500.0
    image = signal * jnp.ones((256, 256))
    tel = TelescopeConfig(exposure_time_s=100.0, read_noise_e=4.0,
                           dark_current_e_s=0.01)
    noisy = add_noise(key, image, tel, "JWST/NIRCam.F200W")
    # Mean should be within a few percent of signal + sky + dark
    sky = sky_background_rate("JWST/NIRCam.F200W", tel.pixel_scale) * tel.exposure_time_s
    dark = tel.dark_current_e_s * tel.exposure_time_s
    expected = signal + sky + dark
    npt.assert_allclose(float(jnp.mean(noisy)), expected, rtol=0.05)


def test_sky_background_rate_positive():
    rate = sky_background_rate("JWST/NIRCam.F200W", 0.031)
    assert rate > 0
