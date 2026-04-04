"""Galaxy population layer.

Generates a catalog of galaxies for a single sky tile by:
1. Computing expected counts per redshift bin from the GSMF + comoving volume
2. Sampling stellar masses from the double-Schechter function
3. Assigning sizes (mass-size), metallicities (MZR), SFH types/params
4. Looking up photometry from the precomputed table
5. Assigning sky positions uniformly within the tile
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from skysim.config import SimConfig
from skysim.coordinates import TileInfo
from skysim.layers.base import Catalog
from skysim.models.mass_metallicity import sample_metallicities
from skysim.models.mass_size import sample_sizes
from skysim.models.photometry import PhotTable, log_lnu_to_apparent_mag
from skysim.models.schechter import (
    expected_count_in_volume,
    sample_masses,
    weaver23_params,
)
from skysim.models.sfh import (
    SFHType,
    assign_age,
    assign_sfh_type,
    assign_tau,
    snap_to_grid,
    TAU_GRID_GYR,
    AGE_GRID_GYR,
)
from skysim.seed import layer_key
from skysim.utils.cosmology import comoving_volume_shell, luminosity_distances

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "seds"
DEFAULT_TABLE = DATA_DIR / "phot_table.npz"


class GalaxyLayer:
    """Galaxy population layer."""

    name = "galaxies"

    def __init__(self, phot_table_path: str | Path | None = None):
        path = Path(phot_table_path) if phot_table_path else DEFAULT_TABLE
        self.phot = PhotTable(path)

    def generate_catalog(
        self,
        key: jax.Array,
        tile: TileInfo,
        config: SimConfig,
    ) -> Catalog:
        """Generate a galaxy catalog for one tile."""
        z_edges = np.array(config.redshift_bin_edges)
        log_m_min = 7.0

        # --- Pass 1: count galaxies per redshift bin ---
        counts_per_bin = []
        for i in range(len(z_edges) - 1):
            z_lo, z_hi = float(z_edges[i]), float(z_edges[i + 1])
            z_mid = 0.5 * (z_lo + z_hi)
            params = weaver23_params(z_mid)
            vol = comoving_volume_shell(z_lo, z_hi, tile.area_arcmin2)
            n_expected = expected_count_in_volume(params, vol, log_m_min=log_m_min)
            counts_per_bin.append(max(0, n_expected))

        # Poisson-sample the counts
        k1, k2 = jax.random.split(key)
        counts = jax.random.poisson(k1, jnp.array(counts_per_bin))
        counts = np.array(counts, dtype=int)
        n_total = int(counts.sum())

        if n_total == 0:
            return _empty_catalog(config.active_filter)

        # --- Pass 2: sample masses and redshifts for all bins ---
        # Pre-draw all uniform random numbers at once, then slice per bin
        k_mass, k_z = jax.random.split(k2)
        u_mass = jax.random.uniform(k_mass, shape=(n_total,))
        u_z = jax.random.uniform(k_z, shape=(n_total,))

        log_mass = jnp.empty(n_total, dtype=jnp.float32)
        z = jnp.empty(n_total, dtype=jnp.float32)
        offset = 0

        for i in range(len(z_edges) - 1):
            n_i = int(counts[i])
            if n_i == 0:
                continue
            z_lo, z_hi = float(z_edges[i]), float(z_edges[i + 1])
            z_mid = 0.5 * (z_lo + z_hi)
            params = weaver23_params(z_mid)

            # Build CDF once per bin and invert the pre-drawn uniforms
            from skysim.models.schechter import _build_cdf
            log_m_grid, cdf = _build_cdf(log_m_min, 12.5, params)
            masses_i = jnp.interp(u_mass[offset:offset + n_i], cdf, log_m_grid)
            log_mass = log_mass.at[offset:offset + n_i].set(masses_i)

            # Uniform redshifts within the bin
            z_i = z_lo + u_z[offset:offset + n_i] * (z_hi - z_lo)
            z = z.at[offset:offset + n_i].set(z_i)
            offset += n_i

        # --- Assign galaxy properties ---
        k3, k4, k5, k6, k7, k8, k_dust = jax.random.split(
            jax.random.fold_in(key, 99), 7
        )

        # Morphological type (late/early split by mass)
        f_late = jax.nn.sigmoid(-2.0 * (log_mass - 10.5 + 0.3 * z))
        is_late_type = jax.random.uniform(k3, shape=log_mass.shape) < f_late

        # Sizes
        log_re = sample_sizes(k4, log_mass, z, is_late_type)

        # Metallicities
        metallicity = sample_metallicities(k5, log_mass, z)

        # SFH
        sfh_type = assign_sfh_type(k6, log_mass, z)
        tau_gyr = assign_tau(k7, log_mass, z, sfh_type)
        age_gyr = assign_age(k8, z)

        # Dust optical depth: correlates with mass and SFR (late-type = dustier)
        tau_v = assign_tau_v(k_dust, log_mass, is_late_type, config.dust)

        # --- Photometry lookup ---
        sfh_idx = sfh_type.astype(jnp.int32)
        tau_grid_idx = snap_to_grid(tau_gyr, TAU_GRID_GYR)
        age_grid_idx = snap_to_grid(age_gyr, AGE_GRID_GYR)

        log_lnu = self.phot.lookup(
            sfh_idx, tau_gyr, age_gyr, metallicity, log_mass,
            config.active_filter,
            tau_v=tau_v,
        )

        # Apparent magnitude
        dl_mpc = jnp.array(luminosity_distances(np.array(z)))
        mag = log_lnu_to_apparent_mag(log_lnu, z, dl_mpc)

        # --- Sky positions: uniform within tile ---
        k9 = jax.random.fold_in(key, 200)
        k_ra, k_dec = jax.random.split(k9)

        # Approximate tile extent (sqrt of area)
        side_deg = jnp.sqrt(tile.area_arcmin2) / 60.0
        ra = tile.ra_center + (jax.random.uniform(k_ra, shape=(n_total,)) - 0.5) * side_deg
        dec = tile.dec_center + (jax.random.uniform(k_dec, shape=(n_total,)) - 0.5) * side_deg

        # Sersic index: early-type → n~4, late-type → n~1
        sersic_n = jnp.where(is_late_type, 1.0, 4.0)
        # Add scatter
        k10 = jax.random.fold_in(key, 300)
        sersic_n = sersic_n + 0.5 * jax.random.normal(k10, shape=log_mass.shape)
        sersic_n = jnp.clip(sersic_n, 0.5, 8.0)

        # Axis ratio
        k11 = jax.random.fold_in(key, 400)
        q = jnp.where(
            is_late_type,
            0.3 + 0.5 * jax.random.uniform(k11, shape=log_mass.shape),
            0.5 + 0.4 * jax.random.uniform(k11, shape=log_mass.shape),
        )

        # Position angle (random)
        k12 = jax.random.fold_in(key, 500)
        pa = jax.random.uniform(k12, shape=(n_total,), minval=0.0, maxval=jnp.pi)

        return {
            "ra": ra,
            "dec": dec,
            "z": z,
            "log_mass": log_mass,
            "log_re_kpc": log_re,
            "metallicity": metallicity,
            "sfh_type": sfh_type,
            "tau_gyr": tau_gyr,
            "age_gyr": age_gyr,
            "tau_v": tau_v,
            "sersic_n": sersic_n,
            "axis_ratio": q,
            "pa": pa,
            "is_late_type": is_late_type,
            "mag": mag,
            "log_lnu": log_lnu,
        }

    def render(
        self,
        catalog: Catalog,
        image: jnp.ndarray,
        config: SimConfig,
    ) -> jnp.ndarray:
        """Render galaxies onto an image (placeholder — Phase 3)."""
        # Phase 3 will implement Sersic profile rendering
        return image


def assign_tau_v(
    key: jax.Array,
    log_mass: jnp.ndarray,
    is_late_type: jnp.ndarray,
    dust_config,
) -> jnp.ndarray:
    """Assign V-band dust optical depth to each galaxy.

    Star-forming (late-type) galaxies are dustier on average.
    tau_v correlates positively with stellar mass.
    """
    tv_min, tv_max = dust_config.tau_v_ism_range

    # Mean tau_v: higher for late-type, scales with mass
    mean_tv = jnp.where(
        is_late_type,
        0.3 + 0.15 * (log_mass - 9.0),   # star-forming: moderate dust
        0.05 + 0.05 * (log_mass - 10.0),  # quiescent: low dust
    )
    mean_tv = jnp.clip(mean_tv, 0.0, tv_max)

    # Log-normal scatter
    scatter = 0.3 * jax.random.normal(key, shape=log_mass.shape)
    tau_v = mean_tv * jnp.exp(scatter)
    tau_v = jnp.clip(tau_v, tv_min, tv_max)
    return tau_v


def _empty_catalog(filter_code: str) -> Catalog:
    """Return an empty catalog with the right column names."""
    return {
        "ra": jnp.array([]),
        "dec": jnp.array([]),
        "z": jnp.array([]),
        "log_mass": jnp.array([]),
        "log_re_kpc": jnp.array([]),
        "metallicity": jnp.array([]),
        "sfh_type": jnp.array([], dtype=jnp.int32),
        "tau_gyr": jnp.array([]),
        "age_gyr": jnp.array([]),
        "tau_v": jnp.array([]),
        "sersic_n": jnp.array([]),
        "axis_ratio": jnp.array([]),
        "pa": jnp.array([]),
        "is_late_type": jnp.array([], dtype=jnp.bool_),
        "mag": jnp.array([]),
        "log_lnu": jnp.array([]),
    }
