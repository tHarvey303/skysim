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
    """Parameters for a double-Schechter function at a single redshift.

    phi(M) dM = exp(-M/M*) * [phi1*(M/M*)^alpha1 + phi2*(M/M*)^alpha2] d(M/M*)

    where M = 10^log_m (stellar mass in solar masses).
    phi1, phi2 are in units of dex^-1 Mpc^-3.
    """
    log_mstar: float   # log10(M*/Msun)
    phi1: float         # normalisation 1 (Mpc^-3 dex^-1)
    alpha1: float       # faint-end slope 1
    phi2: float         # normalisation 2
    alpha2: float       # faint-end slope 2


def weaver23_params(z: float) -> SchechterParams:
    """Weaver+23 COSMOS2020 double-Schechter GSMF parameters.

    Approximate parameterisation of Table 3 in Weaver+23, interpolated
    linearly between their redshift bins. Valid for 0.2 < z < 5.5.
    """
    # Redshift bin centers and best-fit values from Weaver+23 Table 3
    z_bins = jnp.array([0.35, 0.65, 0.95, 1.25, 1.75, 2.25, 2.75, 3.50, 4.50])
    log_mstar_vals = jnp.array([10.79, 10.84, 10.89, 10.93, 10.96, 10.98, 11.02, 11.08, 11.15])
    log_phi1_vals = jnp.array([-2.44, -2.54, -2.61, -2.75, -2.98, -3.23, -3.41, -3.73, -4.18])
    alpha1_vals = jnp.array([-0.28, -0.39, -0.28, -0.41, -0.50, -0.73, -0.68, -0.82, -1.04])
    log_phi2_vals = jnp.array([-3.08, -3.33, -3.55, -3.63, -3.77, -4.20, -4.65, -5.00, -5.50])
    alpha2_vals = jnp.array([-1.48, -1.53, -1.56, -1.59, -1.57, -1.63, -1.76, -1.88, -2.00])

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
