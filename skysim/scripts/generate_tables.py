"""One-time script: build photometry lookup tables using Synthesizer.

Iterates over a grid of (SFH type, tau, age, metallicity, tau_v) and
computes rest-frame luminosities L_nu in each filter per solar mass of
stars formed, including dust attenuation and dust emission via
Synthesizer's BimodalPacmanEmission model with a Calzetti (2000) curve.

Output: data/seds/phot_table.npz

Usage:
    python -m skysim.scripts.generate_tables
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from synthesizer import Grid
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.emission_models.generators.dust.greybody import Greybody
from synthesizer.emission_models.stellar.pacman_model import (
    BimodalPacmanEmission,
)
from synthesizer.instruments import FilterCollection
from synthesizer.parametric import SFH, Stars
from unyt import dimensionless, Gyr, K, Msun

from skysim.models.sfh import AGE_GRID_GYR, TAU_GRID_GYR, SFHType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_DIR = "/Users/user/Documents/PhD/synthesizer/grids/"
GRID_NAME = "test_grid"

FILTER_CODES = [
    # JWST NIRCam
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F444W",
    # HST ACS/WFC3
    "HST/ACS_WFC.F435W",
    "HST/ACS_WFC.F606W",
    "HST/ACS_WFC.F814W",
    "HST/WFC3_IR.F105W",
    "HST/WFC3_IR.F125W",
    "HST/WFC3_IR.F160W",
    # Euclid
    "Euclid/VIS.vis",
    "Euclid/NISP.H",
    "Euclid/NISP.J",
    "Euclid/NISP.Y",
    # Rubin/LSST
    "LSST/LSST.u",
    "LSST/LSST.g",
    "LSST/LSST.r",
    "LSST/LSST.i",
    "LSST/LSST.z",
    "LSST/LSST.y",
    # HSC grizy
    "Subaru/HSC.g",
    "Subaru/HSC.r",
    "Subaru/HSC.i",
    "Subaru/HSC.z",
    "Subaru/HSC.y",
    # Spitzer IRAC
    "Spitzer/IRAC.I1",
    "Spitzer/IRAC.I2",
]

# Discrete tau_v values for the precomputed grid
# (optical depth in the V band for ISM component;
#  birth cloud tau_v is set equal for simplicity)
TAU_V_GRID = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "seds"


# ---------------------------------------------------------------------------
# SFH builders
# ---------------------------------------------------------------------------

def _make_sfh(sfh_type: int, tau_gyr: float, age_gyr: float):
    """Create a Synthesizer SFH object."""
    max_age = age_gyr * Gyr
    if sfh_type == SFHType.CONSTANT:
        return SFH.Constant(max_age=max_age)
    elif sfh_type == SFHType.DECLINING_EXP:
        return SFH.DecliningExponential(tau=tau_gyr * Gyr, max_age=max_age)
    elif sfh_type == SFHType.DELAYED_EXP:
        return SFH.DelayedExponential(tau=tau_gyr * Gyr, max_age=max_age)
    else:
        raise ValueError(f"Unknown SFH type: {sfh_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(verbose: bool = True):
    """Build the photometry lookup table and save to disk."""
    # Load SPS grid
    grid = Grid(GRID_NAME, grid_dir=GRID_DIR)

    # Dust emission model (Greybody for IR re-emission)
    dust_emission = Greybody(temperature=40 * K, emissivity=1.5)

    # Full emission model: stellar + nebular + dust attenuation + dust emission
    # Uses BimodalPacmanEmission which splits into young/old populations
    # with separate birth-cloud and ISM dust components.
    emission_model = BimodalPacmanEmission(
        grid=grid,
        tau_v_ism="tau_v_ism",
        tau_v_birth="tau_v_birth",
        dust_curve_ism=Calzetti2000(),
        dust_curve_birth=Calzetti2000(),
        age_pivot=7 * dimensionless,  # log10(age/yr) = 7 → 10 Myr
        dust_emission_ism=dust_emission,
        dust_emission_birth=dust_emission,
        fesc=0.0,
        fesc_ly_alpha=0.1,
    )

    # Filters
    filters = FilterCollection(filter_codes=FILTER_CODES)

    # Grid axes
    sfh_types = np.array([s.value for s in SFHType])
    tau_vals = np.array(TAU_GRID_GYR)
    age_vals = np.array(AGE_GRID_GYR)
    Z_vals = np.array(grid.metallicities)
    tau_v_vals = TAU_V_GRID

    n_sfh = len(sfh_types)
    n_tau = len(tau_vals)
    n_age = len(age_vals)
    n_Z = len(Z_vals)
    n_tv = len(tau_v_vals)
    n_filt = len(FILTER_CODES)

    total = n_sfh * n_tau * n_age * n_Z * n_tv
    if verbose:
        print(f"Generating photometry table: {n_sfh} SFH × {n_tau} tau × "
              f"{n_age} age × {n_Z} Z × {n_tv} tau_v × {n_filt} filters")
        print(f"  = {total} grid points ({total * n_filt} evaluations)")

    # Output array: log10(L_nu / erg s^-1 Hz^-1) per solar mass
    # Shape: (n_sfh, n_tau, n_age, n_Z, n_tv, n_filters)
    log_lnu = np.full(
        (n_sfh, n_tau, n_age, n_Z, n_tv, n_filt), np.nan, dtype=np.float32
    )

    count = 0
    for i_sfh, i_tau, i_age in itertools.product(
        range(n_sfh), range(n_tau), range(n_age)
    ):
        sfh_type = sfh_types[i_sfh]
        tau_gyr = float(tau_vals[i_tau])
        age_gyr = float(age_vals[i_age])

        # For constant SFH, tau is irrelevant — skip duplicates
        if sfh_type == SFHType.CONSTANT and i_tau > 0:
            log_lnu[i_sfh, i_tau, :, :, :, :] = log_lnu[i_sfh, 0, :, :, :, :]
            continue

        try:
            sfh = _make_sfh(sfh_type, tau_gyr, age_gyr)
        except Exception as e:
            if verbose:
                print(f"  Skipping sfh={sfh_type} tau={tau_gyr} age={age_gyr}: {e}")
            count += n_Z * n_tv
            continue

        for i_Z in range(n_Z):
            Z = float(Z_vals[i_Z])

            for i_tv in range(n_tv):
                tau_v = float(tau_v_vals[i_tv])

                try:
                    stars = Stars(
                        grid.log10ages,
                        grid.metallicities,
                        sf_hist=sfh,
                        metal_dist=Z,
                        initial_mass=1.0 * Msun,
                    )

                    # Set dust optical depths on the stars object
                    stars.tau_v_ism = tau_v
                    stars.tau_v_birth = tau_v

                    stars.get_spectra(emission_model)
                    phot_dict = stars.get_photo_lnu(filters)

                    # Use the "total" key — full attenuated + dust emission
                    phot = phot_dict["total"]

                    for i_f, fc in enumerate(FILTER_CODES):
                        lnu = float(phot[fc])
                        if lnu > 0:
                            log_lnu[i_sfh, i_tau, i_age, i_Z, i_tv, i_f] = (
                                np.log10(lnu)
                            )

                except Exception as e:
                    if verbose and count < 5:
                        print(
                            f"  Error at sfh={sfh_type} tau={tau_gyr} "
                            f"age={age_gyr} Z={Z:.4e} tau_v={tau_v}: {e}"
                        )

                count += 1
                if verbose and count % 500 == 0:
                    print(f"  {count}/{total} done...")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "phot_table.npz"
    np.savez_compressed(
        out_path,
        log_lnu=log_lnu,
        filter_codes=np.array(FILTER_CODES),
        sfh_types=sfh_types,
        tau_gyr=tau_vals,
        age_gyr=age_vals,
        metallicities=Z_vals,
        tau_v=tau_v_vals,
    )

    n_valid = np.isfinite(log_lnu).sum()
    n_total = log_lnu.size
    if verbose:
        print(
            f"Saved {out_path} ({n_valid}/{n_total} valid entries, "
            f"{out_path.stat().st_size / 1e6:.1f} MB)"
        )

    return out_path


if __name__ == "__main__":
    generate()
