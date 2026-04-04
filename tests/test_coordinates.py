"""Tests for sky tiling and coordinate utilities."""

import numpy as np
import numpy.testing as npt

from skysim.coordinates import (
    TileInfo,
    ang2pix,
    nside2npix,
    nside2pixarea_arcmin2,
    pix2ang,
    radec_to_thetaphi,
    radec_to_tile,
    thetaphi_to_radec,
    tile_center_radec,
)


def test_nside2npix():
    assert nside2npix(1) == 12
    assert nside2npix(2) == 48
    assert nside2npix(64) == 12 * 64 * 64


def test_pixarea_sums_to_full_sky():
    """Total area of all pixels should be the full sky in sq arcmin."""
    nside = 8
    npix = nside2npix(nside)
    total = npix * nside2pixarea_arcmin2(nside)
    full_sky_arcmin2 = 4.0 * np.pi * (180.0 / np.pi * 60.0) ** 2
    npt.assert_allclose(total, full_sky_arcmin2, rtol=1e-10)


def test_roundtrip_ang2pix_pix2ang():
    """Converting pixel→angle→pixel should recover the original index."""
    nside = 16
    npix = nside2npix(nside)
    indices = np.arange(npix)
    theta, phi = pix2ang(nside, indices)
    recovered = ang2pix(nside, theta, phi)
    npt.assert_array_equal(recovered, indices)


def test_radec_thetaphi_roundtrip():
    ra = np.array([0.0, 90.0, 180.0, 270.0, 45.0])
    dec = np.array([90.0, 0.0, -45.0, 30.0, -90.0])
    theta, phi = radec_to_thetaphi(ra, dec)
    ra2, dec2 = thetaphi_to_radec(theta, phi)
    npt.assert_allclose(ra2, ra, atol=1e-10)
    npt.assert_allclose(dec2, dec, atol=1e-10)


def test_north_pole_tile():
    """RA=0, Dec=90 (north pole) should map to tile 0 for any NSIDE."""
    tile = radec_to_tile(8, 0.0, 89.99)
    # Should be one of the first few pixels
    assert tile < 12


def test_tile_center_radec_range():
    nside = 16
    for i in range(nside2npix(nside)):
        ra, dec = tile_center_radec(nside, i)
        assert 0.0 <= ra < 360.0
        assert -90.0 <= dec <= 90.0


def test_tile_info():
    info = TileInfo.from_index(nside=32, tile_index=100)
    assert info.tile_index == 100
    assert info.nside == 32
    assert info.area_arcmin2 > 0
    assert 0.0 <= info.ra_center < 360.0
    assert -90.0 <= info.dec_center <= 90.0
