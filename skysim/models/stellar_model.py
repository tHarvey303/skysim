"""Milky Way stellar density model.

Simplified Besancon-like model with four components:
thin disc, thick disc, stellar halo, and bulge.
Gives stellar number density as a function of Galactic coordinates
and absolute magnitude.

Coordinate convention:
- (l, b) Galactic longitude/latitude in degrees
- R: Galactocentric cylindrical radius (kpc)
- z: height above the plane (kpc)
- R_sun = 8.0 kpc, z_sun = 0.025 kpc
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Solar position
R_SUN = 8.0    # kpc
Z_SUN = 0.025  # kpc above the plane


def galactic_to_Rz(
    l_deg: jnp.ndarray,
    b_deg: jnp.ndarray,
    d_kpc: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert Galactic (l, b, distance) to cylindrical (R, z)."""
    l = jnp.deg2rad(l_deg)
    b = jnp.deg2rad(b_deg)

    x = d_kpc * jnp.cos(b) * jnp.cos(l) - R_SUN
    y = d_kpc * jnp.cos(b) * jnp.sin(l)
    z = d_kpc * jnp.sin(b) + Z_SUN

    R = jnp.sqrt(x**2 + y**2)
    return R, z


# ---------------------------------------------------------------------------
# Component densities (stars per kpc^3 per mag)
# ---------------------------------------------------------------------------

def thin_disc_density(R: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Thin disc: double exponential.

    rho ~ exp(-R/h_R) * exp(-|z|/h_z)
    h_R = 2.6 kpc, h_z = 0.3 kpc
    """
    h_R = 2.6
    h_z = 0.3
    rho_0 = 4.0e7  # normalisation: stars/kpc^3 (integrated over LF)
    return rho_0 * jnp.exp(-R / h_R) * jnp.exp(-jnp.abs(z) / h_z)


def thick_disc_density(R: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Thick disc: double exponential with larger scale height.

    h_R = 3.6 kpc, h_z = 0.9 kpc, ~10% of thin disc normalisation.
    """
    h_R = 3.6
    h_z = 0.9
    rho_0 = 4.0e6
    return rho_0 * jnp.exp(-R / h_R) * jnp.exp(-jnp.abs(z) / h_z)


def halo_density(R: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Stellar halo: oblate power-law.

    rho ~ (r_eff)^(-3.4)  where r_eff = sqrt(R^2 + (z/q)^2), q = 0.6
    """
    q = 0.6
    r_eff = jnp.sqrt(R**2 + (z / q) ** 2)
    r_eff = jnp.maximum(r_eff, 0.1)  # avoid singularity
    rho_0 = 1.0e4
    return rho_0 * (r_eff / R_SUN) ** (-3.4)


def bulge_density(R: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Triaxial bulge (simplified as oblate exponential).

    rho ~ exp(-r_s / 0.5) where r_s = sqrt(R^2 + (z/0.4)^2)
    Concentrated within ~2 kpc of the Galactic center.
    """
    r_s = jnp.sqrt(R**2 + (z / 0.4) ** 2)
    rho_0 = 3.0e8
    return rho_0 * jnp.exp(-r_s / 0.5)


def total_density(R: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Total stellar number density (all components)."""
    return (
        thin_disc_density(R, z)
        + thick_disc_density(R, z)
        + halo_density(R, z)
        + bulge_density(R, z)
    )


# ---------------------------------------------------------------------------
# Stellar luminosity function (simplified)
# ---------------------------------------------------------------------------

# Discrete absolute magnitude bins and relative number densities
# (very simplified Kroupa IMF + MS lifetime weighting)
ABS_MAG_BINS = np.array([
    -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
])
# Relative number per mag bin (faint stars dominate)
LF_WEIGHTS = np.array([
    0.001, 0.003, 0.008, 0.02, 0.05, 0.10, 0.15, 0.18,
    0.15, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001,
])
LF_WEIGHTS = LF_WEIGHTS / LF_WEIGHTS.sum()


def expected_star_count(
    l_deg: float,
    b_deg: float,
    area_arcmin2: float,
    mag_limit: float = 28.0,
) -> float:
    """Estimate total star count toward (l, b) above a magnitude limit.

    Integrates the density model along the line of sight.
    """
    area_sr = area_arcmin2 / (180.0 / np.pi * 60.0) ** 2

    # Distance bins (kpc)
    d_edges = np.logspace(-1, 2, 50)  # 0.1 to 100 kpc
    d_mid = 0.5 * (d_edges[:-1] + d_edges[1:])
    dd = np.diff(d_edges)

    l_arr = np.full_like(d_mid, l_deg)
    b_arr = np.full_like(d_mid, b_deg)
    R, z = galactic_to_Rz(jnp.array(l_arr), jnp.array(b_arr), jnp.array(d_mid))
    rho = np.array(total_density(R, z))

    # Volume element: dV = d^2 * dd * area_sr
    dV = d_mid**2 * dd * area_sr  # kpc^3

    # Total stars (all magnitudes)
    n_total = np.sum(rho * dV)

    # Apply magnitude limit: fraction of LF visible at each distance
    # distance modulus = 5*log10(d_kpc) + 10
    dm = 5.0 * np.log10(d_mid) + 10.0
    frac_visible = np.zeros_like(d_mid)
    for i, d in enumerate(d_mid):
        abs_limit = mag_limit - dm[i]
        visible = ABS_MAG_BINS < abs_limit
        frac_visible[i] = LF_WEIGHTS[visible].sum() if visible.any() else 0.0

    return float(np.sum(rho * dV * frac_visible))


def radec_to_lb(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Convert (RA, Dec) to Galactic (l, b) in degrees.

    Uses the standard transformation matrix.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # J2000 NGP: (ra, dec) = (192.8595, 27.1284) deg
    ra_ngp = np.deg2rad(192.8595)
    dec_ngp = np.deg2rad(27.1284)
    l_ncp = np.deg2rad(122.932)

    sdec_ngp = np.sin(dec_ngp)
    cdec_ngp = np.cos(dec_ngp)

    sin_b = sdec_ngp * np.sin(dec) + cdec_ngp * np.cos(dec) * np.cos(ra - ra_ngp)
    b = np.arcsin(np.clip(sin_b, -1, 1))

    y = np.cos(dec) * np.sin(ra - ra_ngp)
    x = cdec_ngp * np.sin(dec) - sdec_ngp * np.cos(dec) * np.cos(ra - ra_ngp)
    l = l_ncp - np.arctan2(y, x)

    return float(np.rad2deg(l)) % 360.0, float(np.rad2deg(b))
