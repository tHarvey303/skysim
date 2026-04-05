"""Double-Schechter galaxy stellar mass function (GSMF).

Implements the Weaver+23 parameterisation with redshift evolution.
Sampling via inverse-CDF on a finely tabulated grid, all in JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass
class SchechterParams:
    """Parameters for a single or double-Schechter function at a single redshift.

    phi(M) dM = exp(-M/M*) * [phi1*(M/M*)^alpha1 + phi2*(M/M*)^alpha2] d(M/M*)

    where M = 10^log_m (stellar mass in solar masses).
    phi1, phi2 are in units of dex^-1 Mpc^-3.
    """
    log_mstar: float   # log10(M*/Msun)
    phi1: float         # normalisation 1 (Mpc^-3 dex^-1)
    alpha1: float       # faint-end slope 1
    phi2: float         # normalisation 2 (or zero for single Schechter)
    alpha2: float       # faint-end slope 2


def weaver23_params(z: float) -> SchechterParams:
    """Weaver+23 COSMOS2020 single/double-Schechter GSMF parameters.

    Approximate parameterisation of Table 3 in Weaver+23, interpolated
    linearly between their redshift bins. Valid for 0.2 < z < 5.5.
    """
    # Redshift bin centers and best-fit values from Weaver+23 Table 3
    # Using the bin midpoints for z_bins
    z_bins: Array = jnp.array([
        0.35, 0.65, 0.95, 1.30, 1.75, 2.25, 2.75, 3.25, 4.00, 5.00, 6.00, 7.00
    ])

    # Extracting the central median likelihood values
    log_mstar_vals: Array = jnp.array([
        10.89, 10.96, 11.02, 11.00, 10.86, 10.78, 10.94, 11.08, 10.65, 10.40, 10.17, 10.23
    ])

    alpha1_vals: Array = jnp.array([
        -1.42, -1.40, -1.32, -1.33, -1.48, -1.45, -1.55, -1.55, -1.55, -1.55, -1.55, -1.55
    ])

    # \Phi values are converted from the table's notation (X * 10^-3) to log10(\Phi)
    log_phi1_vals: Array = jnp.array([
        -3.13, -3.19, -3.07, -3.15, -3.54, -3.57, -3.74, -3.96, -3.92, -4.00, -4.30, -4.70
    ])

    # The second Schechter component is absent for z > 3.0; normalisation phi2 is set to zero and slope alpha2 is not well constrained. Using a very small number for phi2 and a fixed slope to avoid issues with log10(0) and to allow smooth interpolation.   
    alpha2_vals: Array = jnp.array([
        -0.45, -0.63, -0.61, -0.49, -0.42, 0.08, -0.07, -0.05, -0.05, -0.05, -0.05, -0.05
    ])
    # Second component normalisation phi2 is also zero for z > 3.0; using a very small number to avoid issues with log10(0)
    log_phi2_vals: Array = jnp.array([
        -2.97, -3.07, -3.18, -3.44, -3.19, -3.57, -4.30, -99.0, -99.0, -99.0, -99.0, -99.0
    ])
    z_c = jnp.clip(z, z_bins[0], z_bins[-1])
    log_mstar = jnp.interp(z_c, z_bins, log_mstar_vals)
    log_phi1 = jnp.interp(z_c, z_bins, log_phi1_vals)
    alpha1 = jnp.interp(z_c, z_bins, alpha1_vals)
    log_phi2 = jnp.interp(z_c, z_bins, log_phi2_vals)
    alpha2 = jnp.interp(z_c, z_bins, alpha2_vals)

    return SchechterParams(
        log_mstar=float(log_mstar),
        phi1=float(10.0 ** log_phi1),
        alpha1=float(alpha1),
        phi2=float(10.0 ** log_phi2),
        alpha2=float(alpha2),
    )


def double_schechter_phi(log_m: jnp.ndarray, params: SchechterParams) -> jnp.ndarray:
    """Evaluate the double-Schechter function phi(log_m) in dex^-1 Mpc^-3.

    Parameters
    ----------
    log_m : array
        log10(M/Msun) values at which to evaluate.
    params : SchechterParams

    Returns
    -------
    phi : array
        Number density in dex^-1 Mpc^-3.
    """
    x = 10.0 ** (log_m - params.log_mstar)
    ln10 = jnp.log(10.0)
    exp_term = jnp.exp(-x)
    phi = ln10 * exp_term * (
        params.phi1 * x ** (params.alpha1 + 1)
        + params.phi2 * x ** (params.alpha2 + 1)
    )
    return phi


def _build_cdf(
    log_m_min: float,
    log_m_max: float,
    params: SchechterParams,
    n_grid: int = 500,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build a normalised CDF table for inverse-CDF sampling."""
    log_m_grid = jnp.linspace(log_m_min, log_m_max, n_grid)
    phi = double_schechter_phi(log_m_grid, params)
    phi = jnp.maximum(phi, 0.0)
    # Trapezoidal integration for CDF
    dlm = log_m_grid[1] - log_m_grid[0]
    cumulative = jnp.cumsum(0.5 * (phi[:-1] + phi[1:]) * dlm)
    cdf = jnp.concatenate([jnp.array([0.0]), cumulative])
    cdf = cdf / cdf[-1]
    return log_m_grid, cdf


def sample_masses(
    key: jax.Array,
    n: int,
    params: SchechterParams,
    log_m_min: float = 7.0,
    log_m_max: float = 12.5,
) -> jnp.ndarray:
    """Sample stellar masses from the double-Schechter GSMF via inverse CDF.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    n : int
        Number of masses to sample.
    params : SchechterParams
    log_m_min, log_m_max : float
        Mass range in log10(M/Msun).

    Returns
    -------
    log_masses : array of shape (n,)
        Sampled log10(M/Msun) values.
    """
    log_m_grid, cdf = _build_cdf(log_m_min, log_m_max, params)
    u = jax.random.uniform(key, shape=(n,))
    log_masses = jnp.interp(u, cdf, log_m_grid)
    return log_masses


def expected_number_density(
    params: SchechterParams,
    log_m_min: float = 7.0,
    log_m_max: float = 12.5,
    n_grid: int = 2000,
) -> float:
    """Integrate the GSMF to get total number density (Mpc^-3) above log_m_min."""
    log_m_grid = jnp.linspace(log_m_min, log_m_max, n_grid)
    phi = double_schechter_phi(log_m_grid, params)
    phi = jnp.maximum(phi, 0.0)
    dlm = log_m_grid[1] - log_m_grid[0]
    return float(jnp.trapezoid(phi, dx=dlm))


def expected_count_in_volume(
    params: SchechterParams,
    volume_mpc3: float,
    log_m_min: float = 7.0,
    log_m_max: float = 12.5,
) -> float:
    """Expected number of galaxies in a comoving volume."""
    n_density = expected_number_density(params, log_m_min, log_m_max)
    return n_density * volume_mpc3
