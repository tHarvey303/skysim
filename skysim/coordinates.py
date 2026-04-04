"""Sky tiling and coordinate utilities.

Provides HEALPix equal-area tiling so that any tile can be generated
independently. Uses healpy for the core pixel ↔ angle conversions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import healpy as hp
import numpy as np


# ---------------------------------------------------------------------------
# HEALPix utilities
# ---------------------------------------------------------------------------

def nside2npix(nside: int) -> int:
    """Total number of pixels for a given NSIDE."""
    return hp.nside2npix(nside)


def nside2pixarea(nside: int) -> float:
    """Pixel solid angle in steradians."""
    return hp.nside2pixarea(nside)


def nside2pixarea_arcmin2(nside: int) -> float:
    """Pixel solid angle in square arcminutes."""
    return hp.nside2pixarea(nside, degrees=True) * 3600.0


def ang2pix(nside: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Convert (theta, phi) in radians to HEALPix RING pixel index."""
    return hp.ang2pix(nside, theta, phi)


def pix2ang(nside: int, ipix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert HEALPix RING pixel index to (theta, phi) center in radians."""
    return hp.pix2ang(nside, ipix)


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def radec_to_thetaphi(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (RA, Dec) in degrees to HEALPix (theta, phi) in radians."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    theta = np.pi / 2.0 - dec
    phi = ra % (2.0 * np.pi)
    return theta, phi


def thetaphi_to_radec(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert HEALPix (theta, phi) in radians to (RA, Dec) in degrees."""
    dec = np.rad2deg(np.pi / 2.0 - theta)
    ra = np.rad2deg(phi) % 360.0
    return ra, dec


def tile_center_radec(nside: int, tile_index: int) -> Tuple[float, float]:
    """Return the (RA, Dec) center of a HEALPix tile in degrees."""
    theta, phi = pix2ang(nside, np.array([tile_index]))
    ra, dec = thetaphi_to_radec(theta, phi)
    return float(ra[0]), float(dec[0])


def radec_to_tile(nside: int, ra_deg: float, dec_deg: float) -> int:
    """Return the HEALPix tile index containing a given (RA, Dec)."""
    theta, phi = radec_to_thetaphi(np.array([ra_deg]), np.array([dec_deg]))
    return int(ang2pix(nside, theta, phi)[0])


# ---------------------------------------------------------------------------
# Tile field-of-view helpers
# ---------------------------------------------------------------------------

@dataclass
class TileInfo:
    """Metadata for a single sky tile."""
    tile_index: int
    nside: int
    ra_center: float   # degrees
    dec_center: float   # degrees
    area_arcmin2: float

    @classmethod
    def from_index(cls, nside: int, tile_index: int) -> "TileInfo":
        ra, dec = tile_center_radec(nside, tile_index)
        area = nside2pixarea_arcmin2(nside)
        return cls(
            tile_index=tile_index,
            nside=nside,
            ra_center=ra,
            dec_center=dec,
            area_arcmin2=area,
        )
