"""Photometry lookup from precomputed SED tables.

At table-generation time (scripts/generate_tables.py), Synthesizer
computes luminosities on a grid of (sfh_type, tau, age, metallicity, tau_v)
for each filter, including dust attenuation and dust emission.
At runtime, this module loads that grid and interpolates to get apparent
magnitudes for each galaxy.

The grid is stored as an .npz file with:
- "log_lnu": array of shape (n_sfh, n_tau, n_age, n_Z, n_tv, n_filters)
    log10(L_nu / erg s^-1 Hz^-1) per unit solar mass of stars formed.
- "filter_codes": 1-D string array of filter names
- "sfh_types": 1-D int array
- "tau_gyr": 1-D float array
- "age_gyr": 1-D float array
- "metallicities": 1-D float array
- "tau_v": 1-D float array (V-band optical depth)
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class PhotTable:
    """Precomputed photometry lookup table."""

    def __init__(self, path: str | Path):
        data = np.load(path, allow_pickle=True)
        self.log_lnu = jnp.array(data["log_lnu"], dtype=jnp.float32)
        self.filter_codes = list(data["filter_codes"])
        self.sfh_types = jnp.array(data["sfh_types"], dtype=jnp.int32)
        self.log_tau = jnp.log10(jnp.array(data["tau_gyr"], dtype=jnp.float32))
        self.log_age = jnp.log10(jnp.array(data["age_gyr"], dtype=jnp.float32))
        self.log_Z = jnp.log10(jnp.array(data["metallicities"], dtype=jnp.float32))
        self.n_filters = len(self.filter_codes)

        # tau_v axis (may not exist in old tables)
        if "tau_v" in data:
            self.tau_v_grid = jnp.array(data["tau_v"], dtype=jnp.float32)
            self.has_dust = True
        else:
            self.has_dust = False

    def filter_index(self, filter_code: str) -> int:
        return self.filter_codes.index(filter_code)

    def lookup(
        self,
        sfh_idx: jnp.ndarray,
        tau_gyr: jnp.ndarray,
        age_gyr: jnp.ndarray,
        metallicity: jnp.ndarray,
        log_mass: jnp.ndarray,
        filter_code: str,
        tau_v: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Look up log10(L_nu) for a batch of galaxies in one filter.

        Uses nearest-neighbour in all grid axes.

        Parameters
        ----------
        sfh_idx : int array (n,)
            SFH type index.
        tau_gyr : float array (n,)
            e-folding time in Gyr.
        age_gyr : float array (n,)
            Stellar population age in Gyr.
        metallicity : float array (n,)
            Metal mass fraction Z.
        log_mass : float array (n,)
            log10(M/Msun) — used to scale the per-unit-mass luminosity.
        filter_code : str
            Which filter to look up.
        tau_v : float array (n,) or None
            V-band dust optical depth. If None and table has dust axis,
            defaults to 0.0 (no dust).

        Returns
        -------
        log_lnu : float array (n,)
            log10(L_nu / erg s^-1 Hz^-1).
        """
        fi = self.filter_index(filter_code)

        # Find nearest grid indices for tau, age, Z
        tau_idx = _nearest_idx(jnp.log10(tau_gyr), self.log_tau)
        age_idx = _nearest_idx(jnp.log10(age_gyr), self.log_age)
        Z_idx = _nearest_idx(jnp.log10(metallicity), self.log_Z)

        if self.has_dust:
            # Grid shape: (sfh, tau, age, Z, tau_v, filter)
            if tau_v is None:
                tv_idx = jnp.zeros_like(sfh_idx)  # index 0 = tau_v=0
            else:
                tv_idx = _nearest_idx(tau_v, self.tau_v_grid)
            log_lnu_per_msun = self.log_lnu[
                sfh_idx, tau_idx, age_idx, Z_idx, tv_idx, fi
            ]
        else:
            # Old table without dust axis: (sfh, tau, age, Z, filter)
            log_lnu_per_msun = self.log_lnu[sfh_idx, tau_idx, age_idx, Z_idx, fi]

        # Scale by stellar mass
        return log_lnu_per_msun + log_mass


def _nearest_idx(values: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Find index of nearest grid point for each value."""
    diffs = jnp.abs(values[:, None] - grid[None, :])
    return jnp.argmin(diffs, axis=1)


def log_lnu_to_apparent_mag(
    log_lnu: jnp.ndarray,
    z: jnp.ndarray,
    dl_mpc: jnp.ndarray,
) -> jnp.ndarray:
    """Convert rest-frame log10(L_nu) to apparent AB magnitude.

    AB magnitude: m = -2.5 * log10(f_nu) - 48.60
    where f_nu = L_nu * (1+z) / (4 * pi * d_L^2)
    (the (1+z) factor accounts for bandwidth compression).

    Parameters
    ----------
    log_lnu : array
        log10(L_nu / erg s^-1 Hz^-1).
    z : array
        Redshift.
    dl_mpc : array
        Luminosity distance in Mpc.

    Returns
    -------
    mag : array
        Apparent AB magnitude.
    """
    # d_L in cm
    dl_cm = dl_mpc * 3.0856776e24  # Mpc -> cm
    log_fnu = (
        log_lnu
        + jnp.log10(1.0 + z)
        - jnp.log10(4.0 * jnp.pi)
        - 2.0 * jnp.log10(dl_cm)
    )
    mag = -2.5 * log_fnu - 48.60
    return mag
