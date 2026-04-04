"""Tests for the mass-metallicity relation."""

import jax
import jax.numpy as jnp

from skysim.models.mass_metallicity import (
    oh12_to_mass_fraction,
    sample_metallicities,
    zahid14_metallicity,
)


def test_mzr_increases_with_mass():
    """More massive galaxies should be more metal-rich."""
    log_m = jnp.array([8.0, 9.0, 10.0, 11.0])
    z = jnp.array([0.5, 0.5, 0.5, 0.5])
    oh12 = zahid14_metallicity(log_m, z)
    # Should be monotonically increasing
    assert jnp.all(jnp.diff(oh12) > 0)


def test_mzr_saturates():
    """MZR should saturate at high mass (asymptotic Z_0)."""
    log_m = jnp.array([11.0, 12.0])
    z = jnp.array([0.5, 0.5])
    oh12 = zahid14_metallicity(log_m, z)
    # Difference should be small at high mass
    assert float(oh12[1] - oh12[0]) < 0.1


def test_solar_calibration():
    """12 + log(O/H) = 8.69 should give Z ~ 0.0142."""
    Z = oh12_to_mass_fraction(jnp.array(8.69))
    assert abs(float(Z) - 0.0142) < 0.001


def test_sample_metallicities_positive():
    key = jax.random.PRNGKey(0)
    log_m = jnp.full(1000, 10.0)
    z = jnp.full(1000, 1.0)
    Z = sample_metallicities(key, log_m, z)
    assert jnp.all(Z > 0)
