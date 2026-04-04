"""Tests for large-scale structure density field."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from skysim.layers.lss import (
    cdm_power_spectrum,
    density_at_positions,
    generate_gaussian_field,
    growth_factor_approx,
    zeldovich_displacement,
)


def test_power_spectrum_positive():
    k = jnp.logspace(-2, 1, 100)
    Pk = cdm_power_spectrum(k)
    assert jnp.all(Pk > 0)


def test_power_spectrum_shape():
    """P(k) should peak at intermediate k and fall at large k."""
    k = jnp.logspace(-2, 1, 100)
    Pk = cdm_power_spectrum(k)
    peak_idx = int(jnp.argmax(Pk))
    assert 5 < peak_idx < 95  # peak not at extremes


def test_gaussian_field_zero_mean():
    """Gaussian random field should have approximately zero mean."""
    key = jax.random.PRNGKey(0)
    delta = generate_gaussian_field(key, ngrid=32, box_size_mpc=100.0)
    npt.assert_allclose(float(jnp.mean(delta)), 0.0, atol=0.5)


def test_gaussian_field_deterministic():
    key = jax.random.PRNGKey(42)
    d1 = generate_gaussian_field(key, 16, 50.0)
    d2 = generate_gaussian_field(key, 16, 50.0)
    npt.assert_allclose(d1, d2)


def test_zeldovich_density_positive():
    """Density field (1+delta) should be non-negative everywhere."""
    key = jax.random.PRNGKey(0)
    rho = zeldovich_displacement(key, 32, 100.0)
    assert jnp.all(rho >= 0)


def test_zeldovich_mean_near_one():
    """Mean of 1+delta should be close to 1."""
    key = jax.random.PRNGKey(0)
    rho = zeldovich_displacement(key, 32, 100.0)
    npt.assert_allclose(float(jnp.mean(rho)), 1.0, atol=0.3)


def test_growth_factor_decreases_with_z():
    """Growth factor should decrease at higher redshift."""
    D0 = growth_factor_approx(0.0)
    D1 = growth_factor_approx(1.0)
    D3 = growth_factor_approx(3.0)
    assert D0 > D1 > D3


def test_growth_factor_unity_at_z0():
    """D(z=0) should be 1 by normalisation."""
    npt.assert_allclose(growth_factor_approx(0.0), 1.0, atol=1e-10)


def test_density_sampling():
    """Sampling density field should return reasonable values."""
    key = jax.random.PRNGKey(0)
    rho = zeldovich_displacement(key, 16, 50.0)
    x = jnp.array([10.0, 25.0, 40.0])
    y = jnp.array([5.0, 15.0, 30.0])
    z = jnp.array([1.0, 20.0, 45.0])
    vals = density_at_positions(rho, x, y, z, 50.0)
    assert vals.shape == (3,)
    assert jnp.all(vals >= 0)
