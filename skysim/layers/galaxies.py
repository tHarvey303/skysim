"""Galaxy population layer.

Generates a catalog of galaxies for a single sky tile by:
1. Computing expected counts per redshift bin from the GSMF + comoving volume
2. Sampling stellar masses from the double-Schechter function
3. Assigning sizes (mass-size), metallicities (MZR), SFH types/params
4. Assigning bulge-to-total ratio and splitting into bulge+disc components
5. Looking up photometry and applying IGM attenuation (Inoue+2014)
6. Assigning sky positions, optionally modulated by LSS density field
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from skysim.config import SimConfig
from skysim.coordinates import TileInfo
from skysim.layers.base import Catalog
from skysim.models.igm import igm_transmission_filter
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
        density_field: jnp.ndarray | None = None,
        density_box_mpc: float = 200.0,
    ) -> Catalog:
        """Generate a galaxy catalog for one tile.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        tile : TileInfo
            Sky tile.
        config : SimConfig
            Simulation configuration.
        density_field : jnp.ndarray or None
            Optional (ngrid, ngrid, ngrid) LSS density contrast field
            (values are 1+delta). When provided, galaxy positions are
            modulated by the local overdensity via acceptance-rejection.
        density_box_mpc : float
            Side length of the density field box in Mpc/h.
        """
        z_edges = np.array(config.redshift_bin_edges)
        log_m_min = config.mass_function.log_m_min
        log_m_max = config.mass_function.log_m_max

        # --- Pass 1: count galaxies per redshift bin ---
        counts_per_bin = []
        for i in range(len(z_edges) - 1):
            z_lo, z_hi = float(z_edges[i]), float(z_edges[i + 1])
            z_mid = 0.5 * (z_lo + z_hi)
            params = weaver23_params(z_mid)
            vol = comoving_volume_shell(z_lo, z_hi, tile.area_arcmin2)
            n_expected = expected_count_in_volume(params, vol, log_m_min=log_m_min)
            counts_per_bin.append(max(0, n_expected))

        # Poisson-sample the counts (oversample by 2x if LSS modulation active)
        k1, k2 = jax.random.split(key)
        oversample = 2.0 if density_field is not None else 1.0
        counts = jax.random.poisson(k1, jnp.array(counts_per_bin) * oversample)
        counts = np.array(counts, dtype=int)
        n_total = int(counts.sum())

        if n_total == 0:
            return _empty_catalog(config.active_filter)

        # --- Pass 2: sample masses and redshifts for all bins ---
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

            from skysim.models.schechter import _build_cdf
            log_m_grid, cdf = _build_cdf(log_m_min, log_m_max, params)
            masses_i = jnp.interp(u_mass[offset:offset + n_i], cdf, log_m_grid)
            log_mass = log_mass.at[offset:offset + n_i].set(masses_i)

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
        log_re = sample_sizes(k4, log_mass, z, is_late_type, cfg=config.mass_size)

        # Metallicities
        metallicity = sample_metallicities(k5, log_mass, z)

        # SFH
        sfh_type = assign_sfh_type(k6, log_mass, z)
        tau_gyr = assign_tau(k7, log_mass, z, sfh_type)
        age_gyr = assign_age(k8, z)

        # Dust optical depth
        tau_v = assign_tau_v(k_dust, log_mass, is_late_type, config.dust)

        # --- Bulge-to-total ratio ---
        k_bt = jax.random.fold_in(key, 600)
        bulge_to_total = assign_bulge_to_total(k_bt, log_mass, is_late_type)

        # Bulge and disc half-light radii
        # Bulge is compact (~40% of total R_e), disc is extended (~130%)
        log_re_bulge = log_re + jnp.log10(0.4)
        log_re_disc = log_re + jnp.log10(1.3)

        # --- Photometry lookup ---
        sfh_idx = sfh_type.astype(jnp.int32)

        log_lnu = self.phot.lookup(
            sfh_idx, tau_gyr, age_gyr, metallicity, log_mass,
            config.active_filter,
            tau_v=tau_v,
        )

        # Apply IGM attenuation (Inoue+2014)
        t_igm = igm_transmission_filter(z, config.active_filter)
        # Avoid log10(0) for fully absorbed sources
        log_t_igm = jnp.log10(jnp.maximum(t_igm, 1e-30))
        log_lnu = log_lnu + log_t_igm

        # Apparent magnitude
        dl_mpc = jnp.array(luminosity_distances(np.array(z)))
        mag = log_lnu_to_apparent_mag(log_lnu, z, dl_mpc)

        # --- Sky positions: uniform within tile ---
        k9 = jax.random.fold_in(key, 200)
        k_ra, k_dec = jax.random.split(k9)

        side_deg = jnp.sqrt(tile.area_arcmin2) / 60.0
        ra = tile.ra_center + (jax.random.uniform(k_ra, shape=(n_total,)) - 0.5) * side_deg
        dec = tile.dec_center + (jax.random.uniform(k_dec, shape=(n_total,)) - 0.5) * side_deg

        # --- LSS density modulation (acceptance-rejection) ---
        if density_field is not None:
            k_lss = jax.random.fold_in(key, 700)
            keep = _lss_accept_reject(
                k_lss, ra, dec, z, density_field, density_box_mpc,
                tile.ra_center, tile.dec_center,
            )
            # Filter catalog to kept galaxies
            kept_idx = jnp.where(keep, size=int(jnp.sum(keep)), fill_value=0)[0]
            n_kept = len(kept_idx)
            if n_kept == 0:
                return _empty_catalog(config.active_filter)

            ra = ra[kept_idx]
            dec = dec[kept_idx]
            z = z[kept_idx]
            log_mass = log_mass[kept_idx]
            log_re = log_re[kept_idx]
            log_re_bulge = log_re_bulge[kept_idx]
            log_re_disc = log_re_disc[kept_idx]
            metallicity = metallicity[kept_idx]
            sfh_type = sfh_type[kept_idx]
            tau_gyr = tau_gyr[kept_idx]
            age_gyr = age_gyr[kept_idx]
            tau_v = tau_v[kept_idx]
            bulge_to_total = bulge_to_total[kept_idx]
            is_late_type = is_late_type[kept_idx]
            mag = mag[kept_idx]
            log_lnu = log_lnu[kept_idx]
            n_total = n_kept

        # --- Morphological parameters ---
        # Sersic index: early-type → n~4, late-type → n~1 (overall)
        sersic_n = jnp.where(is_late_type, 1.0, 4.0)
        k10 = jax.random.fold_in(key, 300)
        sersic_n = sersic_n + 0.5 * jax.random.normal(k10, shape=(n_total,))
        sersic_n = jnp.clip(sersic_n, 0.5, 8.0)

        # Axis ratio
        k11 = jax.random.fold_in(key, 400)
        q = jnp.where(
            is_late_type,
            0.3 + 0.5 * jax.random.uniform(k11, shape=(n_total,)),
            0.5 + 0.4 * jax.random.uniform(k11, shape=(n_total,)),
        )

        # Bulge axis ratio (rounder than disc)
        q_bulge = jnp.clip(q + 0.2, 0.3, 1.0)

        # Position angle (random)
        k12 = jax.random.fold_in(key, 500)
        pa = jax.random.uniform(k12, shape=(n_total,), minval=0.0, maxval=jnp.pi)

        return {
            "ra": ra,
            "dec": dec,
            "z": z,
            "log_mass": log_mass,
            "log_re_kpc": log_re,
            "log_re_bulge_kpc": log_re_bulge,
            "log_re_disc_kpc": log_re_disc,
            "metallicity": metallicity,
            "sfh_type": sfh_type,
            "tau_gyr": tau_gyr,
            "age_gyr": age_gyr,
            "tau_v": tau_v,
            "bulge_to_total": bulge_to_total,
            "sersic_n": sersic_n,
            "axis_ratio": q,
            "axis_ratio_bulge": q_bulge,
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
        return image


# ---------------------------------------------------------------------------
# Bulge-to-total ratio assignment
# ---------------------------------------------------------------------------

def assign_bulge_to_total(
    key: jax.Array,
    log_mass: jnp.ndarray,
    is_late_type: jnp.ndarray,
) -> jnp.ndarray:
    """Assign bulge-to-total light ratio.

    Late-type galaxies are disc-dominated (B/T ~ 0.1-0.3).
    Early-type galaxies are bulge-dominated (B/T ~ 0.5-0.9).
    Scatter increases at intermediate masses.
    """
    k1, k2 = jax.random.split(key)

    # Late-type: B/T peaks around 0.15, range [0, 0.5]
    bt_late = 0.15 + 0.1 * jax.random.normal(k1, shape=log_mass.shape)
    bt_late = jnp.clip(bt_late, 0.0, 0.5)

    # Early-type: B/T peaks around 0.75, range [0.3, 1.0]
    bt_early = 0.75 + 0.12 * jax.random.normal(k2, shape=log_mass.shape)
    bt_early = jnp.clip(bt_early, 0.3, 1.0)

    return jnp.where(is_late_type, bt_late, bt_early)


# ---------------------------------------------------------------------------
# LSS density modulation
# ---------------------------------------------------------------------------

def _lss_accept_reject(
    key: jax.Array,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    z: jnp.ndarray,
    density_field: jnp.ndarray,
    box_mpc: float,
    ra_center: float,
    dec_center: float,
) -> jnp.ndarray:
    """Accept/reject galaxies based on local LSS overdensity.

    Converts (RA, Dec, z) to comoving coordinates, looks up the
    density field, and accepts each galaxy with probability
    proportional to (1+delta).

    Returns a boolean mask of galaxies to keep.
    """
    from skysim.utils.cosmology import comoving_distances_jax

    # Convert to comoving coordinates (Mpc)
    dc = comoving_distances_jax(z)

    # Transverse comoving coordinates (small-angle approx)
    dra_rad = (ra - ra_center) * (jnp.pi / 180.0) * jnp.cos(jnp.deg2rad(dec))
    ddec_rad = (dec - dec_center) * (jnp.pi / 180.0)
    x_mpc = dc * dra_rad
    y_mpc = dc * ddec_rad
    z_mpc = dc  # line-of-sight

    # Map into periodic box
    ngrid = density_field.shape[0]
    cell = box_mpc / ngrid

    ix = ((x_mpc % box_mpc) / cell).astype(jnp.int32) % ngrid
    iy = ((y_mpc % box_mpc) / cell).astype(jnp.int32) % ngrid
    iz = ((z_mpc % box_mpc) / cell).astype(jnp.int32) % ngrid

    # Look up density (1+delta)
    density = density_field[ix, iy, iz]

    # Normalise: accept with probability density / max(density)
    # Since we oversampled by 2x, target acceptance rate ≈ 0.5
    max_density = jnp.max(density_field)
    prob = density / max_density

    u = jax.random.uniform(key, shape=ra.shape)
    return u < prob


# ---------------------------------------------------------------------------
# Dust assignment
# ---------------------------------------------------------------------------

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

    mean_tv = jnp.where(
        is_late_type,
        0.3 + 0.15 * (log_mass - 9.0),
        0.05 + 0.05 * (log_mass - 10.0),
    )
    mean_tv = jnp.clip(mean_tv, 0.0, tv_max)

    scatter = 0.3 * jax.random.normal(key, shape=log_mass.shape)
    tau_v = mean_tv * jnp.exp(scatter)
    tau_v = jnp.clip(tau_v, tv_min, tv_max)
    return tau_v


# ---------------------------------------------------------------------------
# Empty catalog
# ---------------------------------------------------------------------------

def _empty_catalog(filter_code: str) -> Catalog:
    """Return an empty catalog with the right column names."""
    return {
        "ra": jnp.array([]),
        "dec": jnp.array([]),
        "z": jnp.array([]),
        "log_mass": jnp.array([]),
        "log_re_kpc": jnp.array([]),
        "log_re_bulge_kpc": jnp.array([]),
        "log_re_disc_kpc": jnp.array([]),
        "metallicity": jnp.array([]),
        "sfh_type": jnp.array([], dtype=jnp.int32),
        "tau_gyr": jnp.array([]),
        "age_gyr": jnp.array([]),
        "tau_v": jnp.array([]),
        "bulge_to_total": jnp.array([]),
        "sersic_n": jnp.array([]),
        "axis_ratio": jnp.array([]),
        "axis_ratio_bulge": jnp.array([]),
        "pa": jnp.array([]),
        "is_late_type": jnp.array([], dtype=jnp.bool_),
        "mag": jnp.array([]),
        "log_lnu": jnp.array([]),
    }
