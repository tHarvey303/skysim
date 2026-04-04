"""Parametric star-formation history assignment.

Assigns SFH type and parameters (tau, age) to galaxies based on their
stellar mass, redshift, and morphological type. The actual SFH
computation is delegated to Synthesizer at table-generation time;
at runtime we only need the SFH *index* into the precomputed grid.

SFH models used:
- Declining exponential: SFR ~ exp(-t/tau)   [quiescent / early-type]
- Delayed exponential:   SFR ~ t*exp(-t/tau)  [star-forming / late-type]
- Constant:              SFR = const           [young starbursts]
"""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp


class SFHType(IntEnum):
    """Integer codes for SFH types — used as grid indices."""
    CONSTANT = 0
    DECLINING_EXP = 1
    DELAYED_EXP = 2


# Discrete tau values (Gyr) for the precomputed grid
TAU_GRID_GYR = jnp.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])

# Discrete max_age values (Gyr) — age of the stellar population
AGE_GRID_GYR = jnp.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0])


def assign_sfh_type(
    key: jax.Array,
    log_mass: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """Assign SFH type based on mass and redshift.

    Simple model: the quiescent fraction increases with mass and
    decreases with redshift, following a sigmoid.

    Returns
    -------
    sfh_type : int array
        SFHType codes (0, 1, or 2).
    """
    # Quiescent fraction: sigmoid in log_mass, shifted by redshift
    f_quiescent = jax.nn.sigmoid(2.0 * (log_mass - 10.5 + 0.5 * z))

    u = jax.random.uniform(key, shape=log_mass.shape)
    is_quiescent = u < f_quiescent

    # Quiescent → declining exp, star-forming ��� delayed exp
    # Small fraction of low-mass star-forming get constant SFH
    key2 = jax.random.fold_in(key, 1)
    u2 = jax.random.uniform(key2, shape=log_mass.shape)
    is_constant = (~is_quiescent) & (u2 < 0.1) & (log_mass < 9.0)

    sfh_type = jnp.where(
        is_quiescent,
        SFHType.DECLINING_EXP,
        jnp.where(is_constant, SFHType.CONSTANT, SFHType.DELAYED_EXP),
    )
    return sfh_type


def assign_tau(
    key: jax.Array,
    log_mass: jnp.ndarray,
    z: jnp.ndarray,
    sfh_type: jnp.ndarray,
) -> jnp.ndarray:
    """Assign e-folding timescale tau (Gyr).

    tau correlates with mass: more massive → shorter tau (faster quenching).
    Star-forming galaxies get longer tau.
    """
    # Base tau from mass (in log Gyr)
    log_tau_base = jnp.where(
        sfh_type == SFHType.DECLINING_EXP,
        -0.3 * (log_mass - 10.0) + jnp.log10(0.5),  # quiescent: short tau
        -0.2 * (log_mass - 10.0) + jnp.log10(2.0),   # star-forming: longer tau
    )
    scatter = 0.2 * jax.random.normal(key, shape=log_mass.shape)
    log_tau = log_tau_base + scatter

    tau_gyr = 10.0 ** log_tau
    tau_gyr = jnp.clip(tau_gyr, TAU_GRID_GYR[0], TAU_GRID_GYR[-1])
    return tau_gyr


def assign_age(
    key: jax.Array,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """Assign stellar population age (Gyr).

    Age is drawn uniformly between 0.1 Gyr and the age of the universe
    at redshift z (approximated).
    """
    # Approximate age of universe at redshift z (flat LCDM, H0=70, Om=0.3)
    # t(z) ≈ 13.8 / (1+z)^1.5 * correction — simplified
    t_universe = 13.8 / jnp.sqrt(1.0 + z) * jnp.sqrt(0.3 * (1.0 + z) ** 3 + 0.7) ** (-1.0 / 1.5)
    # More practically: use Hubble time approximation
    t_universe = jnp.minimum(t_universe, 13.0)
    t_universe = jnp.maximum(t_universe, 0.2)

    u = jax.random.uniform(key, shape=z.shape)
    age_gyr = 0.1 + u * (t_universe - 0.1)
    age_gyr = jnp.clip(age_gyr, AGE_GRID_GYR[0], AGE_GRID_GYR[-1])
    return age_gyr


def snap_to_grid(values: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Snap continuous values to the nearest grid point (returns indices)."""
    diffs = jnp.abs(values[:, None] - grid[None, :])
    return jnp.argmin(diffs, axis=1)
