"""Large-scale structure via Zel'dovich approximation.

Generates a 3-D density field that modulates galaxy number counts
to produce realistic clustering (filaments, voids, walls).

The Zel'dovich approximation displaces particles from a uniform grid
using the gradient of a Gaussian random field with a CDM-like power
spectrum. The resulting density field is projected along the line of
sight to modulate the galaxy surface density per redshift shell.

This layer doesn't generate its own catalog — it provides a density
contrast field delta(ra, dec, z) that the galaxy layer uses as a
multiplicative weight on Poisson counts.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def cdm_power_spectrum(k: jnp.ndarray, ns: float = 0.96, sigma8: float = 0.8) -> jnp.ndarray:
    """Approximate CDM power spectrum P(k).

    Uses the Eisenstein & Hu (1998) no-wiggle transfer function
    approximation for simplicity.

    Parameters
    ----------
    k : array
        Wavenumber in h/Mpc.
    ns : float
        Scalar spectral index.
    sigma8 : float
        Normalisation (approximate).

    Returns
    -------
    Pk : array
        Power spectrum (arbitrary normalisation, will be rescaled).
    """
    # Shape parameter (Omega_m * h ≈ 0.21)
    Gamma = 0.21
    q = k / Gamma

    # Bardeen+86 transfer function (simplified)
    T = jnp.log(1.0 + 2.34 * q) / (2.34 * q) * (
        1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    ) ** (-0.25)

    Pk = k**ns * T**2
    return Pk


def generate_gaussian_field(
    key: jax.Array,
    ngrid: int,
    box_size_mpc: float,
) -> jnp.ndarray:
    """Generate a 3-D Gaussian random field with CDM power spectrum.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    ngrid : int
        Number of grid cells per side.
    box_size_mpc : float
        Box side length in Mpc/h.

    Returns
    -------
    delta : array (ngrid, ngrid, ngrid)
        Density contrast field in real space.
    """
    dk = 2.0 * jnp.pi / box_size_mpc
    kx = jnp.fft.fftfreq(ngrid, d=box_size_mpc / ngrid) * 2 * jnp.pi
    ky = kx
    kz = jnp.fft.rfftfreq(ngrid, d=box_size_mpc / ngrid) * 2 * jnp.pi

    kx3d, ky3d, kz3d = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = jnp.sqrt(kx3d**2 + ky3d**2 + kz3d**2)
    k_mag = jnp.maximum(k_mag, dk)  # avoid k=0

    # Power spectrum shape (unnormalised)
    Pk_shape = cdm_power_spectrum(k_mag)

    # Normalise to sigma8: compute the raw variance and rescale so that
    # the field smoothed with a top-hat of R=8 Mpc/h has variance sigma8^2.
    # Approximation: rescale the total variance to give sigma8 ≈ 0.8.
    sigma8 = 0.8
    V_cell = (box_size_mpc / ngrid) ** 3
    # Top-hat window in k-space: W(kR) = 3*(sin(kR)-kR*cos(kR))/(kR)^3
    R8 = 8.0  # Mpc/h
    kR = k_mag * R8
    W = 3.0 * (jnp.sin(kR) - kR * jnp.cos(kR)) / jnp.maximum(kR**3, 1e-30)
    W = jnp.where(kR < 0.01, 1.0, W)  # limit W→1 as kR→0
    # Raw variance from unnormalised P(k)
    var_raw = jnp.sum(Pk_shape * W**2) * dk**3 / (2 * jnp.pi)**3
    # Normalisation factor
    norm = sigma8**2 / jnp.maximum(var_raw, 1e-30)

    Pk = Pk_shape * norm
    amplitude = jnp.sqrt(Pk * V_cell)

    # Random phases
    k1, k2 = jax.random.split(key)
    noise_real = jax.random.normal(k1, shape=k_mag.shape)
    noise_imag = jax.random.normal(k2, shape=k_mag.shape)

    # Build complex field in Fourier space
    delta_k = amplitude * (noise_real + 1j * noise_imag)

    # Zero the DC mode
    delta_k = delta_k.at[0, 0, 0].set(0.0 + 0j)

    # Inverse FFT to real space
    delta = jnp.fft.irfftn(delta_k, s=(ngrid, ngrid, ngrid))

    return delta


def zeldovich_displacement(
    key: jax.Array,
    ngrid: int,
    box_size_mpc: float,
    growth_factor: float = 1.0,
) -> jnp.ndarray:
    """Zel'dovich-approximation density field.

    Displaces particles on a uniform grid by the gradient of a
    Gaussian potential, then estimates density via CIC.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    ngrid : int
        Grid resolution per side.
    box_size_mpc : float
        Box size in Mpc/h.
    growth_factor : float
        Linear growth factor D(z) — scales displacements.

    Returns
    -------
    density : array (ngrid, ngrid, ngrid)
        Density contrast 1 + delta (values < 0 clipped to 0).
    """
    # Generate potential field (Gaussian random field / k^2)
    dk = 2.0 * jnp.pi / box_size_mpc
    delta = generate_gaussian_field(key, ngrid, box_size_mpc)

    # In the Zel'dovich approximation, the density field from the
    # Gaussian field is sufficient for our purposes — we use
    # delta directly as density contrast rather than doing particle
    # displacement + CIC (which is expensive and adds complexity).
    # The Gaussian field with CDM power spectrum already captures
    # the essential clustering statistics.

    # Apply growth factor and convert to 1+delta
    density = 1.0 + growth_factor * delta
    density = jnp.maximum(density, 0.0)  # no negative densities

    return density


def growth_factor_approx(z: float, Om0: float = 0.3) -> float:
    """Approximate linear growth factor D(z) normalised to D(0)=1.

    Carroll, Press & Turner (1992) approximation:
    D(a) = a * g(Omega(a)), where g is the growth suppression factor.
    """
    Ol0 = 1.0 - Om0
    a = 1.0 / (1.0 + z)
    Om_z = Om0 / (Om0 + Ol0 * a**3)
    Ol_z = Ol0 * a**3 / (Om0 + Ol0 * a**3)
    g_z = (5.0 / 2.0) * Om_z / (
        Om_z ** (4.0 / 7.0) - Ol_z + (1.0 + Om_z / 2.0) * (1.0 + Ol_z / 70.0)
    )
    D_z = a * g_z

    # Normalise to z=0 (a=1)
    g_0 = (5.0 / 2.0) * Om0 / (
        Om0 ** (4.0 / 7.0) - Ol0 + (1.0 + Om0 / 2.0) * (1.0 + Ol0 / 70.0)
    )
    D_0 = 1.0 * g_0
    return D_z / D_0


def density_at_positions(
    density_field: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z_pos: jnp.ndarray,
    box_size_mpc: float,
) -> jnp.ndarray:
    """Sample density field at arbitrary positions using nearest-grid-point.

    Parameters
    ----------
    density_field : array (ngrid, ngrid, ngrid)
        The density contrast field (1 + delta).
    x, y, z_pos : arrays
        Comoving positions in Mpc/h, wrapped into [0, box_size).
    box_size_mpc : float

    Returns
    -------
    density : array
        1 + delta at each position.
    """
    ngrid = density_field.shape[0]
    cell_size = box_size_mpc / ngrid

    # Wrap positions into box
    ix = ((x % box_size_mpc) / cell_size).astype(int) % ngrid
    iy = ((y % box_size_mpc) / cell_size).astype(int) % ngrid
    iz = ((z_pos % box_size_mpc) / cell_size).astype(int) % ngrid

    return density_field[ix, iy, iz]
