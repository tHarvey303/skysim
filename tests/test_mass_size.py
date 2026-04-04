"""Tests for the mass-size relation."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from skysim.models.mass_size import log_re_mean, sample_sizes


def test_more_massive_galaxies_are_larger():
    """At fixed z, higher mass → larger R_e."""
    z = jnp.array([1.0, 1.0])
    log_m = jnp.array([9.0, 11.0])
    is_late = jnp.array([True, True])
    log_re = log_re_mean(log_m, z, is_late)
    assert float(log_re[1]) > float(log_re[0])


def test_early_type_smaller_than_late():
    """At fixed mass and z, early-type should be smaller than late-type (at low mass)."""
    z = jnp.array([1.0, 1.0])
    log_m = jnp.array([9.5, 9.5])
    re_late = log_re_mean(log_m, z, jnp.array([True, True]))
    re_early = log_re_mean(log_m, z, jnp.array([False, False]))
    assert float(re_late[0]) > float(re_early[0])


def test_sizes_higher_z_are_smaller():
    """At fixed mass, galaxies at higher z should be smaller."""
    log_m = jnp.array([10.5, 10.5])
    z_lo = jnp.array([0.5, 0.5])
    z_hi = jnp.array([2.5, 2.5])
    is_late = jnp.array([True, True])
    re_lo = log_re_mean(log_m, z_lo, is_late)
    re_hi = log_re_mean(log_m, z_hi, is_late)
    assert float(re_lo[0]) > float(re_hi[0])


def test_sample_sizes_scatter():
    """Sampled sizes should have scatter around the mean."""
    key = jax.random.PRNGKey(0)
    n = 5000
    log_m = jnp.full(n, 10.0)
    z = jnp.full(n, 1.0)
    is_late = jnp.ones(n, dtype=bool)
    log_re = sample_sizes(key, log_m, z, is_late, scatter_dex=0.15)
    mean = log_re_mean(log_m, z, is_late)
    # Standard deviation should be close to 0.15
    npt.assert_allclose(float(jnp.std(log_re)), 0.15, atol=0.02)
