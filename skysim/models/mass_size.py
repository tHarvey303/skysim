"""Mass-size relation following van der Wel+14.

Assigns half-light radii to galaxies based on stellar mass, redshift,
and morphological type (early/late), with log-normal scatter.

Parameters can be overridden via ``MassSizeConfig`` in the simulation
config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from skysim.config import MassSizeConfig


def log_re_mean(
    log_mass: jnp.ndarray,
    z: jnp.ndarray,
    is_late_type: jnp.ndarray,
    cfg: "MassSizeConfig | None" = None,
) -> jnp.ndarray:
    """Mean log10(R_e / kpc) from van der Wel+14 Eq. 2.

    log10(R_e) = log10(A) + beta * (log10(M) - mass_pivot)

    where A and beta depend on morphological type and evolve with redshift.

    Parameters
    ----------
    log_mass : array
        log10(M/Msun).
    z : array
        Redshift.
    is_late_type : array (bool or 0/1)
        True for late-type (disc-dominated), False for early-type.
    cfg : MassSizeConfig, optional
        Override default van der Wel+14 parameters.

    Returns
    -------
    log_re : array
        log10(R_e / kpc).
    """
    if cfg is None:
        from skysim.config import MassSizeConfig
        cfg = MassSizeConfig()

    # Late-type
    log_A_late = jnp.log10(cfg.A0_late) + cfg.gamma_late * jnp.log10(1.0 + z)
    beta_late = cfg.beta_late * jnp.ones_like(z)

    # Early-type
    log_A_early = jnp.log10(cfg.A0_early) + cfg.gamma_early * jnp.log10(1.0 + z)
    beta_early = cfg.beta_early * jnp.ones_like(z)

    log_A = jnp.where(is_late_type, log_A_late, log_A_early)
    beta = jnp.where(is_late_type, beta_late, beta_early)

    log_re = log_A + beta * (log_mass - cfg.mass_pivot)
    return log_re


def sample_sizes(
    key: jax.Array,
    log_mass: jnp.ndarray,
    z: jnp.ndarray,
    is_late_type: jnp.ndarray,
    scatter_dex: float | None = None,
    cfg: "MassSizeConfig | None" = None,
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
    scatter_dex : float, optional
        Log-normal scatter in dex. If None, uses ``cfg.scatter_dex``.
    cfg : MassSizeConfig, optional
        Override default parameters.

    Returns
    -------
    log_re : array
        log10(R_e / kpc) with scatter applied.
    """
    if cfg is None:
        from skysim.config import MassSizeConfig
        cfg = MassSizeConfig()
    if scatter_dex is None:
        scatter_dex = cfg.scatter_dex

    mean = log_re_mean(log_mass, z, is_late_type, cfg=cfg)
    noise = scatter_dex * jax.random.normal(key, shape=log_mass.shape)
    return mean + noise
