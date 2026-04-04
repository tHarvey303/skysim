"""Milky Way stellar foreground layer.

Generates a catalog of foreground stars for a single sky tile based on
the simplified Besancon-like density model, assigns absolute magnitudes
from the luminosity function, and converts to apparent magnitudes.
Stars are rendered as point sources.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from skysim.config import SimConfig
from skysim.coordinates import TileInfo
from skysim.layers.base import Catalog
from skysim.models.stellar_model import (
    ABS_MAG_BINS,
    LF_WEIGHTS,
    R_SUN,
    expected_star_count,
    galactic_to_Rz,
    radec_to_lb,
    total_density,
)


# Approximate colour (V-band abs mag → filter correction)
# Positive means fainter in the filter than in V
FILTER_CORRECTIONS = {
    "JWST/NIRCam.F090W": -0.5,
    "JWST/NIRCam.F115W": -0.8,
    "JWST/NIRCam.F150W": -1.2,
    "JWST/NIRCam.F200W": -1.5,
    "JWST/NIRCam.F277W": -1.8,
    "JWST/NIRCam.F356W": -2.0,
    "JWST/NIRCam.F444W": -2.2,
    "HST/ACS_WFC.F435W": 0.1,
    "HST/ACS_WFC.F606W": -0.1,
    "HST/ACS_WFC.F814W": -0.5,
    "HST/WFC3_IR.F105W": -0.7,
    "HST/WFC3_IR.F125W": -1.0,
    "HST/WFC3_IR.F160W": -1.3,
}


class StarLayer:
    """Milky Way stellar foreground layer."""

    name = "stars"

    def generate_catalog(
        self,
        key: jax.Array,
        tile: TileInfo,
        config: SimConfig,
    ) -> Catalog:
        """Generate a star catalog for one tile."""
        l, b = radec_to_lb(tile.ra_center, tile.dec_center)

        # Expected total star count
        n_expected = expected_star_count(l, b, tile.area_arcmin2, mag_limit=30.0)
        n_expected = max(0, n_expected)

        # Poisson sample
        k1, k2 = jax.random.split(key)
        n = int(jax.random.poisson(k1, jnp.array(n_expected)))
        n = min(n, 500000)  # safety cap

        if n == 0:
            return _empty_star_catalog()

        # --- Sample distances along the line of sight ---
        k_d, k_m, k_ra, k_dec = jax.random.split(k2, 4)

        # Distance sampling: weight by rho(d) * d^2
        d_edges = jnp.logspace(-1, 2, 100)  # 0.1 to 100 kpc
        d_mid = 0.5 * (d_edges[:-1] + d_edges[1:])
        dd = jnp.diff(d_edges)

        l_arr = jnp.full_like(d_mid, l)
        b_arr = jnp.full_like(d_mid, b)
        R, z = galactic_to_Rz(l_arr, b_arr, d_mid)
        rho = total_density(R, z)
        weights = rho * d_mid**2 * dd
        weights = jnp.maximum(weights, 0.0)
        weights = weights / jnp.sum(weights)

        # Sample distance bin indices
        dist_idx = jax.random.choice(k_d, len(d_mid), shape=(n,), p=weights)
        # Add uniform scatter within each bin
        k_scatter = jax.random.fold_in(k_d, 1)
        frac = jax.random.uniform(k_scatter, shape=(n,))
        distances = d_mid[dist_idx] + frac * dd[dist_idx]

        # --- Sample absolute magnitudes from LF ---
        abs_mag_idx = jax.random.choice(
            k_m, len(ABS_MAG_BINS), shape=(n,),
            p=jnp.array(LF_WEIGHTS),
        )
        # Add 0.5 mag scatter within each bin
        k_mag_scatter = jax.random.fold_in(k_m, 1)
        abs_mag = jnp.array(ABS_MAG_BINS)[abs_mag_idx] + \
                  jax.random.uniform(k_mag_scatter, shape=(n,), minval=-0.5, maxval=0.5)

        # Distance modulus → apparent magnitude
        dm = 5.0 * jnp.log10(distances) + 10.0
        filter_corr = FILTER_CORRECTIONS.get(config.active_filter, 0.0)
        app_mag = abs_mag + dm + filter_corr

        # --- Sky positions: uniform within tile ---
        side_deg = jnp.sqrt(tile.area_arcmin2) / 60.0
        ra = tile.ra_center + (jax.random.uniform(k_ra, shape=(n,)) - 0.5) * side_deg
        dec = tile.dec_center + (jax.random.uniform(k_dec, shape=(n,)) - 0.5) * side_deg

        # Convert apparent mag to log_lnu for rendering
        # m_AB = -2.5*log10(f_nu) - 48.60
        log_fnu = -(app_mag + 48.60) / 2.5
        # f_nu to L_nu: L_nu = f_nu * 4*pi*d_L^2 / (1+z)  [z~0 for MW stars]
        # For rendering purposes, we store log_fnu directly

        return {
            "ra": ra,
            "dec": dec,
            "distance_kpc": distances,
            "abs_mag": abs_mag,
            "mag": app_mag,
            "log_fnu": log_fnu,
        }

    def render(
        self,
        catalog: Catalog,
        image: jnp.ndarray,
        config: SimConfig,
    ) -> jnp.ndarray:
        """Render stars as point sources (Phase 3 integration)."""
        # Stars are always point sources — PSF convolution handles the rest
        return image


def _empty_star_catalog() -> Catalog:
    return {
        "ra": jnp.array([]),
        "dec": jnp.array([]),
        "distance_kpc": jnp.array([]),
        "abs_mag": jnp.array([]),
        "mag": jnp.array([]),
        "log_fnu": jnp.array([]),
    }
