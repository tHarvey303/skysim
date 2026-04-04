"""Tests for the double-Schechter GSMF."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from skysim.models.schechter import (
    SchechterParams,
    double_schechter_phi,
    expected_number_density,
    sample_masses,
    weaver23_params,
)


def test_phi_positive():
    """phi should be non-negative everywhere."""
    params = weaver23_params(1.0)
    log_m = jnp.linspace(7.0, 12.5, 200)
    phi = double_schechter_phi(log_m, params)
    assert jnp.all(phi >= 0)


def test_phi_decreases_at_high_mass():
    """Exponential cutoff means phi drops at high mass."""
    params = weaver23_params(1.0)
    phi_10 = double_schechter_phi(jnp.array(10.0), params)
    phi_12 = double_schechter_phi(jnp.array(12.0), params)
    assert float(phi_10) > float(phi_12)


def test_weaver23_redshift_evolution():
    """Phi* should decrease (fewer galaxies) at higher z."""
    p_low = weaver23_params(0.5)
    p_high = weaver23_params(3.0)
    assert p_low.phi1 > p_high.phi1


def test_sample_masses_range():
    key = jax.random.PRNGKey(0)
    params = weaver23_params(1.0)
    log_m = sample_masses(key, 10000, params, log_m_min=8.0, log_m_max=12.0)
    assert jnp.all(log_m >= 8.0)
    assert jnp.all(log_m <= 12.0)


def test_sample_masses_deterministic():
    key = jax.random.PRNGKey(42)
    params = weaver23_params(1.0)
    a = sample_masses(key, 100, params)
    b = sample_masses(key, 100, params)
    npt.assert_allclose(a, b)


def test_number_density_reasonable():
    """Total number density at z~1 should be ~0.01-0.1 Mpc^-3 above 10^8."""
    params = weaver23_params(1.0)
    n = expected_number_density(params, log_m_min=8.0)
    assert 1e-3 < n < 1.0, f"n_density = {n}"
