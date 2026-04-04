"""Mass-size relation following van der Wel+14.

Assigns half-light radii to galaxies based on stellar mass, redshift,
and morphological type (early/late), with log-normal scatter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def log_re_mean(
    log_mass: jnp.ndarray,
    z: jnp.ndarray,
    is_late_type: jnp.ndarray,
) -> jnp.ndarray:
    """Mean log10(R_e / kpc) from van der Wel+14 Eq. 2.

    log10(R_e) = log10(A) + beta * (log10(M) - 10.7)

    where A and beta depend on morphological type and evolve with redshift.

    Parameters
    ----------
    log_mass : array
        log10(M/Msun).
    z : array
        Redshift.
    is_late_type : array (bool or 0/1)
        True for late-type (disc-dominated), False for early-type.

    Returns
    -------
    log_re : array
        log10(R_e / kpc).
    """
    # van der Wel+14 Table 1: A(z) = A0 * (1+z)^gamma, beta = const
    # Late-type
    log_A_late = jnp.log10(6.13) + (-0.37) * jnp.log10(1.0 + z)
    beta_late = 0.22 * jnp.ones_like(z)

    # Early-type
    log_A_early = jnp.log10(4.22) + (-0.70) * jnp.log10(1.0 + z)
    beta_early = 0.76 * jnp.ones_like(z)

    log_A = jnp.where(is_late_type, log_A_late, log_A_early)
    beta = jnp.where(is_late_type, beta_late, beta_early)

    log_re = log_A + beta * (log_mass - 10.7)
    return log_re


def sample_sizes(
    key: jax.Array,
    log_mass: jnp.ndarray,
    z: jnp.ndarray,
    is_late_type: jnp.ndarray,
    scatter_dex: float = 0.15,
) -> jnp.ndarray:
    """Sample half-light radii with log-normal scatter.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    log_mass : array
        log10(M/Msun).
    z : array
        Redshift.
    is_late_type : array (bool or 0/1)
        Morphological type flag.
    scatter_dex : float
        Log-normal scatter in dex (default 0.15 per van der Wel+14).

    Returns
    -------
    log_re : array
        log10(R_e / kpc) with scatter applied.
    """
    mean = log_re_mean(log_mass, z, is_late_type)
    noise = scatter_dex * jax.random.normal(key, shape=log_mass.shape)
    return mean + noise
