"""Inoue+2014 mean IGM attenuation model.

Computes the mean intergalactic medium transmission as a function of
observed wavelength and source redshift, following Inoue, Shimizu,
Iwata & Tanaka (2014, MNRAS 442, 1805).

The model includes:
- Lyman-alpha forest (LAF) line absorption (Lyman series j=2..31)
- Damped Lyman-alpha (DLA) line absorption
- Lyman-continuum (LyC) absorption from LAF and DLA systems

At module load time a filter-averaged transmission table T(z, filter)
is precomputed for fast runtime lookup.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Lyman series wavelengths (Angstrom) and coefficients from Inoue+2014 Table 2
# ---------------------------------------------------------------------------

_LYMAN_LIMIT = 911.75  # Angstrom

# Upper quantum numbers j=2 (Ly-alpha) through j=32
_J_MAX = 31
_J = np.arange(2, _J_MAX + 1)  # j=2..31 → 30 lines
_LAM_J = _LYMAN_LIMIT / (1.0 - 1.0 / _J**2)  # rest wavelengths

# LAF coefficients A^LAF_{1,j} (Table 2 of Inoue+2014)
# These are for the (1+z)^1.2 regime (z < 1.2)
_A_LAF1 = np.array([
    1.690e-2, 4.692e-3, 2.239e-3, 1.319e-3, 8.707e-4,  # j=2-6
    6.178e-4, 4.609e-4, 3.569e-4, 2.843e-4, 2.318e-4,  # j=7-11
    1.923e-4, 1.622e-4, 1.385e-4, 1.196e-4, 1.043e-4,  # j=12-16
    9.174e-5, 8.128e-5, 7.251e-5, 6.505e-5, 5.868e-5,  # j=17-21
    5.319e-5, 4.843e-5, 4.427e-5, 4.063e-5, 3.738e-5,  # j=22-26
    3.454e-5, 3.199e-5, 2.971e-5, 2.766e-5, 2.582e-5,  # j=27-31
])

# DLA coefficients A^DLA_{1,j}
_A_DLA1 = np.array([
    1.617e-4, 1.545e-4, 1.498e-4, 1.460e-4, 1.429e-4,  # j=2-6
    1.402e-4, 1.381e-4, 1.364e-4, 1.349e-4, 1.338e-4,  # j=7-11
    1.328e-4, 1.320e-4, 1.314e-4, 1.309e-4, 1.304e-4,  # j=12-16
    1.300e-4, 1.297e-4, 1.293e-4, 1.291e-4, 1.289e-4,  # j=17-21
    1.287e-4, 1.285e-4, 1.284e-4, 1.283e-4, 1.281e-4,  # j=22-26
    1.280e-4, 1.279e-4, 1.278e-4, 1.278e-4, 1.277e-4,  # j=27-31
])

# Redshift breaks
_Z_LAF1, _Z_LAF2 = 1.2, 4.7   # LAF regime transitions
_Z_DLA1 = 2.0                   # DLA regime transition

# LAF exponents in each regime
_G_LAF = (1.2, 3.7, 5.5)
# DLA exponents
_G_DLA = (2.0, 3.0)

# Derived LAF coefficients for regime continuity
_A_LAF2 = _A_LAF1 * (1 + _Z_LAF1) ** (_G_LAF[0] - _G_LAF[1])
_A_LAF3 = _A_LAF2 * (1 + _Z_LAF2) ** (_G_LAF[1] - _G_LAF[2])

# Derived DLA coefficients
_A_DLA2 = _A_DLA1 * (1 + _Z_DLA1) ** (_G_DLA[0] - _G_DLA[1])

# Lyman continuum coefficients
_A_LAF_LC1 = 2.354e-2
_A_LAF_LC2 = _A_LAF_LC1 * (1 + _Z_LAF1) ** (_G_LAF[0] - _G_LAF[1])
_A_LAF_LC3 = _A_LAF_LC2 * (1 + _Z_LAF2) ** (_G_LAF[1] - _G_LAF[2])

_A_DLA_LC1 = 1.624e-4
_A_DLA_LC2 = _A_DLA_LC1 * (1 + _Z_DLA1) ** (_G_DLA[0] - _G_DLA[1])


# ---------------------------------------------------------------------------
# Optical depth computation
# ---------------------------------------------------------------------------

def _tau_laf_line(lam_obs: np.ndarray, z_s: float) -> np.ndarray:
    """LAF optical depth from Lyman series lines."""
    tau = np.zeros_like(lam_obs)
    for k in range(len(_LAM_J)):
        lj = _LAM_J[k]
        # Only contributes where lambda_obs < lambda_j * (1 + z_s)
        max_lam = lj * (1.0 + z_s)
        mask = lam_obs < max_lam

        ratio = lam_obs / lj
        # Regime 1: lam_obs < lj * (1 + z_LAF1)
        m1 = mask & (lam_obs < lj * (1 + _Z_LAF1))
        tau[m1] += _A_LAF1[k] * ratio[m1] ** _G_LAF[0]

        # Regime 2: lj*(1+z1) <= lam_obs < lj*(1+z2)
        m2 = mask & (~m1) & (lam_obs < lj * (1 + _Z_LAF2))
        tau[m2] += _A_LAF2[k] * ratio[m2] ** _G_LAF[1]

        # Regime 3: lam_obs >= lj*(1+z2)
        m3 = mask & (~m1) & (~m2)
        tau[m3] += _A_LAF3[k] * ratio[m3] ** _G_LAF[2]

    return tau


def _tau_dla_line(lam_obs: np.ndarray, z_s: float) -> np.ndarray:
    """DLA optical depth from Lyman series lines."""
    tau = np.zeros_like(lam_obs)
    for k in range(len(_LAM_J)):
        lj = _LAM_J[k]
        max_lam = lj * (1.0 + z_s)
        mask = lam_obs < max_lam

        ratio = lam_obs / lj
        m1 = mask & (lam_obs < lj * (1 + _Z_DLA1))
        tau[m1] += _A_DLA1[k] * ratio[m1] ** _G_DLA[0]

        m2 = mask & (~m1)
        tau[m2] += _A_DLA2[k] * ratio[m2] ** _G_DLA[1]

    return tau


def _tau_laf_lc(lam_obs: np.ndarray, z_s: float) -> np.ndarray:
    """LAF Lyman-continuum optical depth."""
    tau = np.zeros_like(lam_obs)
    ll = _LYMAN_LIMIT
    max_lam = ll * (1.0 + z_s)
    mask = lam_obs < max_lam

    ratio = lam_obs / ll

    m1 = mask & (lam_obs < ll * (1 + _Z_LAF1))
    tau[m1] += _A_LAF_LC1 * ratio[m1] ** 1.2

    m2 = mask & (~m1) & (lam_obs < ll * (1 + _Z_LAF2))
    tau[m2] += _A_LAF_LC2 * ratio[m2] ** 3.7

    m3 = mask & (~m1) & (~m2)
    tau[m3] += _A_LAF_LC3 * ratio[m3] ** 5.5

    return tau


def _tau_dla_lc(lam_obs: np.ndarray, z_s: float) -> np.ndarray:
    """DLA Lyman-continuum optical depth."""
    tau = np.zeros_like(lam_obs)
    ll = _LYMAN_LIMIT
    max_lam = ll * (1.0 + z_s)
    mask = lam_obs < max_lam

    ratio = lam_obs / ll

    m1 = mask & (lam_obs < ll * (1 + _Z_DLA1))
    tau[m1] += _A_DLA_LC1 * ratio[m1] ** 2.0

    m2 = mask & (~m1)
    tau[m2] += _A_DLA_LC2 * ratio[m2] ** 3.0

    return tau


def inoue14_tau(lam_obs: np.ndarray, z_s: float) -> np.ndarray:
    """Total IGM optical depth at observed wavelength(s) for source at z_s.

    Parameters
    ----------
    lam_obs : array
        Observed wavelength in Angstrom.
    z_s : float
        Source redshift.

    Returns
    -------
    tau : array
        Mean IGM optical depth (>= 0).
    """
    tau = (
        _tau_laf_line(lam_obs, z_s)
        + _tau_dla_line(lam_obs, z_s)
        + _tau_laf_lc(lam_obs, z_s)
        + _tau_dla_lc(lam_obs, z_s)
    )
    return np.maximum(tau, 0.0)


def inoue14_transmission(lam_obs: np.ndarray, z_s: float) -> np.ndarray:
    """Mean IGM transmission exp(-tau) at observed wavelength(s).

    Parameters
    ----------
    lam_obs : array
        Observed wavelength in Angstrom.
    z_s : float
        Source redshift.

    Returns
    -------
    T : array
        Transmission in [0, 1].
    """
    return np.exp(-inoue14_tau(lam_obs, z_s))


# ---------------------------------------------------------------------------
# Filter-averaged IGM transmission table
# ---------------------------------------------------------------------------

# Approximate filter pivot wavelengths (Angstrom) and FWHM
_FILTER_INFO = {
    "JWST/NIRCam.F090W":  (9000, 2000),
    "JWST/NIRCam.F115W":  (11500, 2500),
    "JWST/NIRCam.F150W":  (15000, 3200),
    "JWST/NIRCam.F200W":  (20000, 4600),
    "JWST/NIRCam.F277W":  (27700, 7200),
    "JWST/NIRCam.F356W":  (35600, 7800),
    "JWST/NIRCam.F444W":  (44000, 10800),
    "HST/ACS_WFC.F435W":  (4350, 1000),
    "HST/ACS_WFC.F606W":  (6000, 2500),
    "HST/ACS_WFC.F814W":  (8100, 2500),
    "HST/WFC3_IR.F105W":  (10500, 2700),
    "HST/WFC3_IR.F125W":  (12500, 3000),
    "HST/WFC3_IR.F160W":  (15400, 2700),
    "Euclid/VIS.vis":     (7000, 3500),
    "Euclid/NISP.H":      (17500, 5200),
    "Euclid/NISP.J":      (13500, 4200),
    "Euclid/NISP.Y":      (10800, 2900),
    "LSST/LSST.u":        (3670, 600),
    "LSST/LSST.g":        (4830, 1400),
    "LSST/LSST.r":        (6240, 1400),
    "LSST/LSST.i":        (7540, 1300),
    "LSST/LSST.z":        (8690, 1050),
    "LSST/LSST.y":        (10100, 1100),
}

# Precomputed table: T_IGM(z, filter)
_IGM_Z_GRID = np.linspace(0.0, 8.0, 200)
_IGM_TABLE: dict[str, np.ndarray] = {}  # filter_code → T(z) array


def _build_igm_table():
    """Precompute filter-averaged IGM transmission for all filters and redshifts."""
    for fcode, (pivot, fwhm) in _FILTER_INFO.items():
        # Sample filter bandpass on a fine grid (assume top-hat)
        lam_lo = max(300.0, pivot - fwhm)
        lam_hi = pivot + fwhm
        lam_grid = np.linspace(lam_lo, lam_hi, 200)

        t_arr = np.ones(len(_IGM_Z_GRID), dtype=np.float32)
        for iz, zs in enumerate(_IGM_Z_GRID):
            if zs < 0.01:
                continue
            # Only compute if Ly-alpha at this redshift falls within/below the filter
            lya_obs = 1215.67 * (1 + zs)
            if lya_obs < lam_lo:
                # Entire filter is redward of Ly-alpha: no absorption
                continue
            trans = inoue14_transmission(lam_grid, zs)
            # Filter-averaged transmission (assuming flat SED within band)
            t_arr[iz] = float(np.mean(trans))

        _IGM_TABLE[fcode] = t_arr


# Build table on import
_build_igm_table()

# JAX versions for fast interpolation
_IGM_Z_GRID_J = jnp.array(_IGM_Z_GRID, dtype=jnp.float32)
_IGM_TABLE_J: dict[str, jnp.ndarray] = {
    k: jnp.array(v, dtype=jnp.float32) for k, v in _IGM_TABLE.items()
}


def igm_transmission_filter(z: jnp.ndarray, filter_code: str) -> jnp.ndarray:
    """Look up filter-averaged IGM transmission for an array of redshifts.

    Parameters
    ----------
    z : array
        Source redshifts.
    filter_code : str
        Filter code (must be in the precomputed table).

    Returns
    -------
    T : array
        Mean IGM transmission in [0, 1] for each source.
    """
    if filter_code not in _IGM_TABLE_J:
        return jnp.ones_like(z)
    table = _IGM_TABLE_J[filter_code]
    return jnp.interp(z, _IGM_Z_GRID_J, table)
