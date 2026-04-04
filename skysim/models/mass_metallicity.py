"""Mass-metallicity relation (MZR).

Implements the Zahid+14 universal metallicity relation with redshift
evolution of the turnover mass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def zahid14_metallicity(
    log_mass: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """Gas-phase metallicity 12 + log(O/H) from Zahid+14.

    The model is:
        Z(M, z) = Z_0 + log10(1 - exp(-[M / M_0(z)]^gamma))

    where M_0(z) evolves with redshift and Z_0, gamma are universal.

    Parameters
    ----------
    log_mass : array
        log10(M/Msun).
    z : array
        Redshift.

    Returns
    -------
    oh12 : array
        12 + log10(O/H).
    """
    # Zahid+14 best-fit universal parameters
    Z_0 = 9.102       # asymptotic metallicity
    gamma = 0.513      # power-law slope

    # Turnover mass evolves with redshift: log M_0(z)
    # Zahid+14 Table 2 (approximate fit)
    log_M0 = 9.138 + 0.57 * jnp.log10(1.0 + z)

    x = 10.0 ** (log_mass - log_M0)
    oh12 = Z_0 + jnp.log10(1.0 - jnp.exp(-(x ** gamma)))
    return oh12


def oh12_to_mass_fraction(oh12: jnp.ndarray) -> jnp.ndarray:
    """Convert 12 + log(O/H) to total metal mass fraction Z.

    Uses the solar calibration: 12 + log(O/H)_sun = 8.69, Z_sun = 0.0142.
    Assumes Z scales linearly with O/H.
    """
    Z_sun = 0.0142
    oh12_sun = 8.69
    return Z_sun * 10.0 ** (oh12 - oh12_sun)


def sample_metallicities(
    key: jax.Array,
    log_mass: jnp.ndarray,
    z: jnp.ndarray,
    scatter_dex: float = 0.1,
) -> jnp.ndarray:
    """Sample metal mass fractions Z with scatter around the MZR.

    Returns Z (mass fraction), not 12+log(O/H), since that's what
    SPS models use directly.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    log_mass : array
        log10(M/Msun).
    z : array
        Redshift.
    scatter_dex : float
        Log-normal scatter in 12+log(O/H) (default 0.1 dex).

    Returns
    -------
    Z : array
        Metal mass fraction (dimensionless).
    """
    oh12 = zahid14_metallicity(log_mass, z)
    noise = scatter_dex * jax.random.normal(key, shape=log_mass.shape)
    oh12_scattered = oh12 + noise
    return oh12_to_mass_fraction(oh12_scattered)
