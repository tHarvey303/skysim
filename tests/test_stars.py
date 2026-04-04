"""Tests for the stellar foreground model and layer."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from skysim.models.stellar_model import (
    expected_star_count,
    galactic_to_Rz,
    halo_density,
    radec_to_lb,
    thin_disc_density,
    total_density,
)
from skysim.layers.stars import StarLayer
from skysim.config import SimConfig
from skysim.coordinates import TileInfo


def test_thin_disc_peaks_at_plane():
    """Thin disc density should peak at z=0."""
    R = jnp.array([R_SUN := 8.0, 8.0, 8.0])
    z = jnp.array([0.0, 0.5, 2.0])
    rho = thin_disc_density(R, z)
    assert float(rho[0]) > float(rho[1]) > float(rho[2])


def test_density_higher_toward_center():
    """Total density should be higher toward the Galactic center."""
    z = jnp.array([0.0, 0.0])
    R = jnp.array([2.0, 12.0])
    rho = total_density(R, z)
    assert float(rho[0]) > float(rho[1])


def test_halo_falls_off():
    """Halo density should fall with distance."""
    R = jnp.array([1.0, 10.0, 50.0])
    z = jnp.array([0.0, 0.0, 0.0])
    rho = halo_density(R, z)
    assert float(rho[0]) > float(rho[1]) > float(rho[2])


def test_galactic_to_Rz_at_sun():
    """At d=0, we should recover R_sun, z_sun."""
    R, z = galactic_to_Rz(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]))
    npt.assert_allclose(float(R[0]), 8.0, atol=0.01)
    npt.assert_allclose(float(z[0]), 0.025, atol=0.01)


def test_more_stars_toward_plane():
    """Should see more stars looking toward b=0 than b=90."""
    n_plane = expected_star_count(0.0, 5.0, 100.0, mag_limit=25.0)
    n_pole = expected_star_count(0.0, 85.0, 100.0, mag_limit=25.0)
    assert n_plane > n_pole


def test_radec_to_lb_galactic_center():
    """Galactic center ≈ (RA=266.4, Dec=-28.9) should map to l≈0, b≈0."""
    l, b = radec_to_lb(266.4, -28.9)
    assert abs(b) < 2.0  # should be near the plane
    # l should be near 0 or 360
    assert l < 5.0 or l > 355.0


def test_star_layer_generates_catalog():
    """StarLayer should produce a non-empty catalog for a typical tile."""
    config = SimConfig(active_filter="JWST/NIRCam.F200W")
    tile = TileInfo.from_index(nside=64, tile_index=1000)
    layer = StarLayer()
    key = jax.random.PRNGKey(0)
    catalog = layer.generate_catalog(key, tile, config)
    assert len(catalog["ra"]) > 0
    assert "mag" in catalog
    assert "distance_kpc" in catalog


def test_star_layer_deterministic():
    """Same key → same catalog."""
    config = SimConfig(active_filter="JWST/NIRCam.F200W")
    tile = TileInfo.from_index(nside=64, tile_index=500)
    layer = StarLayer()
    key = jax.random.PRNGKey(42)
    cat1 = layer.generate_catalog(key, tile, config)
    cat2 = layer.generate_catalog(key, tile, config)
    npt.assert_array_equal(cat1["mag"], cat2["mag"])
