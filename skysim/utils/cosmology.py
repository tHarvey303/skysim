"""Lightweight cosmology utilities for comoving volume and distances.

Uses astropy.cosmology for accuracy, with precomputed lookup tables
for fast per-object interpolation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# Default cosmology (Planck18-like)
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)

# ---------------------------------------------------------------------------
# Precomputed lookup tables for fast interpolation
# ---------------------------------------------------------------------------

_Z_TABLE = np.linspace(0.0, 10.0, 5000)
_DL_TABLE = COSMO.luminosity_distance(_Z_TABLE).value.astype(np.float32)  # Mpc
_DA_TABLE = COSMO.angular_diameter_distance(_Z_TABLE).value.astype(np.float32)  # Mpc
_DC_TABLE = COSMO.comoving_distance(_Z_TABLE).value.astype(np.float32)  # Mpc

# JAX versions for fast interpolation
_Z_TABLE_J = jnp.array(_Z_TABLE, dtype=jnp.float32)
_DL_TABLE_J = jnp.array(_DL_TABLE, dtype=jnp.float32)
_DA_TABLE_J = jnp.array(_DA_TABLE, dtype=jnp.float32)


def comoving_volume_between(z_lo: float, z_hi: float) -> float:
    """Comoving volume between two redshifts in Mpc^3, full sky."""
    v_hi = COSMO.comoving_volume(z_hi).value
    v_lo = COSMO.comoving_volume(z_lo).value
    return v_hi - v_lo


def comoving_volume_shell(z_lo: float, z_hi: float, area_arcmin2: float) -> float:
    """Comoving volume of a shell over a given sky area, in Mpc^3."""
    full_sky_arcmin2 = 4.0 * np.pi * (180.0 / np.pi * 60.0) ** 2
    frac = area_arcmin2 / full_sky_arcmin2
    return frac * comoving_volume_between(z_lo, z_hi)


def luminosity_distance(z: float) -> float:
    """Luminosity distance in Mpc."""
    return float(np.interp(z, _Z_TABLE, _DL_TABLE))


def luminosity_distances(z: np.ndarray) -> np.ndarray:
    """Luminosity distances in Mpc for an array of redshifts (fast)."""
    return np.interp(z, _Z_TABLE, _DL_TABLE)


def luminosity_distances_jax(z: jnp.ndarray) -> jnp.ndarray:
    """Luminosity distances in Mpc using JAX interpolation."""
    return jnp.interp(z, _Z_TABLE_J, _DL_TABLE_J)


def angular_diameter_distances(z: np.ndarray) -> np.ndarray:
    """Angular diameter distances in Mpc (fast)."""
    return np.interp(z, _Z_TABLE, _DA_TABLE)


def angular_diameter_distances_jax(z: jnp.ndarray) -> jnp.ndarray:
    """Angular diameter distances in Mpc using JAX interpolation."""
    return jnp.interp(z, _Z_TABLE_J, _DA_TABLE_J)


_DC_TABLE_J = jnp.array(_DC_TABLE, dtype=jnp.float32)


def comoving_distances_jax(z: jnp.ndarray) -> jnp.ndarray:
    """Comoving distances in Mpc using JAX interpolation."""
    return jnp.interp(z, _Z_TABLE_J, _DC_TABLE_J)
