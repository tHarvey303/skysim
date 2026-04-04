"""Tests for Sersic profile rendering."""

import jax.numpy as jnp
import numpy.testing as npt

from skysim.models.morphology import (
    add_stamp_to_image,
    make_sersic_stamp,
    sersic_bn,
    sersic_profile,
    stamp_size_for_galaxy,
)


def test_sersic_bn_n1():
    """b_n for n=1 (exponential) should be ~1.678."""
    bn = sersic_bn(jnp.array(1.0))
    npt.assert_allclose(float(bn), 1.678, atol=0.01)


def test_sersic_bn_n4():
    """b_n for n=4 (de Vaucouleurs) should be ~7.669."""
    bn = sersic_bn(jnp.array(4.0))
    npt.assert_allclose(float(bn), 7.669, atol=0.02)


def test_sersic_profile_flux_conservation():
    """Total flux from numerical integration should match input."""
    r = jnp.linspace(0.01, 50.0, 5000)
    I = sersic_profile(r, r_e=5.0, n=2.0, total_flux=100.0)
    # Integrate: 2*pi*r*I(r)*dr
    dr = r[1] - r[0]
    flux = float(jnp.sum(2.0 * jnp.pi * r * I * dr))
    npt.assert_allclose(flux, 100.0, rtol=0.05)


def test_stamp_sums_to_total_flux():
    """2-D stamp should conserve total flux."""
    stamp = make_sersic_stamp(n=1.0, r_e_pix=5.0, q=1.0, pa=0.0,
                               total_flux=1000.0, stamp_size=61)
    npt.assert_allclose(float(jnp.sum(stamp)), 1000.0, rtol=0.05)


def test_stamp_is_centered():
    """Peak of a circular stamp should be at the center."""
    stamp = make_sersic_stamp(n=4.0, r_e_pix=5.0, q=1.0, pa=0.0,
                               total_flux=1.0, stamp_size=31)
    peak = jnp.unravel_index(jnp.argmax(stamp), stamp.shape)
    assert abs(int(peak[0]) - 15) <= 1
    assert abs(int(peak[1]) - 15) <= 1


def test_elliptical_stamp():
    """An elongated stamp (q < 1) should be wider in one axis."""
    stamp = make_sersic_stamp(n=1.0, r_e_pix=8.0, q=0.3, pa=0.0,
                               total_flux=1.0, stamp_size=61)
    # Sum along rows vs columns — one should be more concentrated
    row_profile = jnp.sum(stamp, axis=1)
    col_profile = jnp.sum(stamp, axis=0)
    row_std = float(jnp.sqrt(jnp.average(jnp.arange(61)**2, weights=row_profile) -
                              jnp.average(jnp.arange(61), weights=row_profile)**2))
    col_std = float(jnp.sqrt(jnp.average(jnp.arange(61)**2, weights=col_profile) -
                              jnp.average(jnp.arange(61), weights=col_profile)**2))
    assert row_std != col_std  # should differ for q != 1


def test_stamp_size_scales_with_re():
    s_small = stamp_size_for_galaxy(2.0, 1.0)
    s_large = stamp_size_for_galaxy(20.0, 1.0)
    assert s_large > s_small


def test_add_stamp_to_image():
    """Stamp should add flux to the correct region."""
    image = jnp.zeros((100, 100), dtype=jnp.float32)
    stamp = jnp.ones((5, 5), dtype=jnp.float32)
    image = add_stamp_to_image(image, stamp, 50, 50)
    assert float(jnp.sum(image)) == 25.0
    assert float(image[50, 50]) == 1.0


def test_add_stamp_boundary_clipping():
    """Stamp at edge should be clipped without error."""
    image = jnp.zeros((100, 100), dtype=jnp.float32)
    stamp = jnp.ones((11, 11), dtype=jnp.float32)
    image = add_stamp_to_image(image, stamp, 2, 2)
    # Should have added partial stamp
    total = float(jnp.sum(image))
    assert 0 < total < 121.0
