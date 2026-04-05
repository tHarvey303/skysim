"""Dataclass-based configuration for SkySim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FilterConfig:
    """A single photometric filter."""
    name: str
    instrument: str
    pivot_wavelength_um: float  # microns
    fwhm_um: float              # microns


@dataclass
class TelescopeConfig:
    """Telescope and instrument parameters."""
    name: str = "default"
    pixel_scale: float = 0.1          # arcsec/pixel
    fov_arcmin: float = 1.0           # field of view side length
    aperture_m: float = 6.5           # primary mirror diameter
    read_noise_e: float = 4.0         # electrons
    dark_current_e_s: float = 0.005   # electrons/s
    exposure_time_s: float = 1000.0   # seconds
    filters: List[FilterConfig] = field(default_factory=list)

    @property
    def npix(self) -> int:
        """Number of pixels along one side of the image."""
        return int(self.fov_arcmin * 60.0 / self.pixel_scale)

    @property
    def image_shape(self) -> Tuple[int, int]:
        return (self.npix, self.npix)


@dataclass
class DustConfig:
    """Dust attenuation model configuration.

    Controls the dust curve and parameters used in the Synthesizer
    emission model for photometry table generation and galaxy property
    assignment at runtime.
    """
    model: str = "calzetti"             # dust curve model name
    tau_v_ism_range: Tuple[float, float] = (0.0, 2.0)   # ISM optical depth range
    tau_v_birth_range: Tuple[float, float] = (0.0, 2.0)  # birth cloud optical depth range
    fesc: float = 0.0                   # escape fraction of ionising photons
    fesc_ly_alpha: float = 0.1          # Lyman-alpha escape fraction
    dust_temp_K: float = 40.0           # greybody dust temperature
    dust_emissivity: float = 1.5        # greybody emissivity index
    age_pivot_log10yr: float = 7.0      # log10(age/yr) separating young/old pops


@dataclass
class MassSizeConfig:
    """Mass-size relation parameters (van der Wel+14 style).

    log10(R_e) = log10(A0 * (1+z)^gamma) + beta * (log10(M) - mass_pivot)
    """
    # Late-type (disc-dominated)
    A0_late: float = 6.13
    gamma_late: float = -0.37
    beta_late: float = 0.22
    # Early-type (elliptical/bulge-dominated)
    A0_early: float = 4.22
    gamma_early: float = -0.70
    beta_early: float = 0.76
    # Common
    mass_pivot: float = 10.7       # log10(M/Msun) pivot mass
    scatter_dex: float = 0.15      # log-normal scatter


@dataclass
class SchechterConfig:
    """Schechter mass function configuration.

    Controls the mass range used for sampling and the model used
    for parameters. The actual Schechter parameters (phi*, M*, alpha)
    are looked up per redshift bin from the chosen model.
    """
    model: str = "weaver23"        # parameter set: "weaver23" (only option for now)
    log_m_min: float = 7.0         # minimum log10(M/Msun)
    log_m_max: float = 12.5        # maximum log10(M/Msun)


@dataclass
class SimConfig:
    """Top-level simulation configuration."""
    seed: int = 42
    nside: int = 64                         # HEALPix NSIDE
    z_min: float = 0.0
    z_max: float = 6.0
    n_redshift_bins: int = 60
    telescope: TelescopeConfig = field(default_factory=TelescopeConfig)
    dust: DustConfig = field(default_factory=DustConfig)
    layers: List[str] = field(default_factory=lambda: ["galaxies", "stars"])
    active_filter: str = "JWST/NIRCam.F200W"
    max_resolved_galaxies: int = 20000
    mass_function: SchechterConfig = field(default_factory=SchechterConfig)
    mass_size: MassSizeConfig = field(default_factory=MassSizeConfig)

    @property
    def redshift_bin_edges(self):
        """Linearly spaced redshift bin edges."""
        import jax.numpy as jnp
        return jnp.linspace(self.z_min, self.z_max, self.n_redshift_bins + 1)


# ---------------------------------------------------------------------------
# Pre-defined telescope presets
# ---------------------------------------------------------------------------

JWST_NIRCAM = TelescopeConfig(
    name="JWST/NIRCam",
    pixel_scale=0.031,
    fov_arcmin=2.2,
    aperture_m=6.5,
    read_noise_e=6.0,
    dark_current_e_s=0.003,
    exposure_time_s=36000.0,
)

HST_ACS = TelescopeConfig(
    name="HST/ACS",
    pixel_scale=0.05,
    fov_arcmin=3.4,
    aperture_m=2.4,
    read_noise_e=4.2,
    dark_current_e_s=0.006,
    exposure_time_s=2000.0,
)

RUBIN_LSST = TelescopeConfig(
    name="Rubin/LSST",
    pixel_scale=0.2,
    fov_arcmin=210.0,
    aperture_m=8.4,
    read_noise_e=9.0,
    dark_current_e_s=0.2,
    exposure_time_s=30.0,
)

EUCLID_VIS = TelescopeConfig(
    name="Euclid/VIS",
    pixel_scale=0.1,
    fov_arcmin=36.0,
    aperture_m=1.2,
    read_noise_e=4.5,
    dark_current_e_s=0.001,
    exposure_time_s=565.0,
)
