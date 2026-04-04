"""Tests for configuration dataclasses."""

from skysim.config import JWST_NIRCAM, SimConfig, TelescopeConfig


def test_default_config():
    cfg = SimConfig()
    assert cfg.seed == 42
    assert cfg.nside == 64
    assert len(cfg.layers) == 2


def test_telescope_npix():
    t = TelescopeConfig(pixel_scale=0.1, fov_arcmin=1.0)
    assert t.npix == 600  # 1 arcmin * 60 arcsec / 0.1 arcsec/pix
    assert t.image_shape == (600, 600)


def test_jwst_preset():
    assert JWST_NIRCAM.pixel_scale == 0.031
    assert JWST_NIRCAM.npix > 0


def test_redshift_bins():
    cfg = SimConfig(z_min=0.0, z_max=3.0, n_redshift_bins=30)
    edges = cfg.redshift_bin_edges
    assert edges.shape == (31,)
    assert float(edges[0]) == 0.0
    assert float(edges[-1]) == 3.0
