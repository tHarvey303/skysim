"""Large-scale structure via Zel'dovich approximation.

Generates a 3-D density field that modulates galaxy number counts
to produce realistic clustering (filaments, voids, walls).

The Zel'dovich approximation (ZA) displaces particles from a uniform
Lagrangian grid using the gradient of the gravitational potential:

    Ψ(k) = −i k / k²  δ(k)          [displacement field]
    x_final = q + D(z) Ψ(q)          [Lagrangian → Eulerian]

Density is then estimated on the Eulerian grid via Nearest-Grid-Point
(NGP) mass assignment.  This produces the characteristic ZA structure —
voids fully emptied, sharp walls and filaments — which Linear Eulerian
Perturbation Theory (density = 1 + D·δ) cannot reproduce.

The projected field δ(ra, dec, z) is used by the galaxy layer as a
multiplicative acceptance-rejection weight on Poisson counts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def cdm_power_spectrum(k: jnp.ndarray, ns: float = 0.96) -> jnp.ndarray:
    """Approximate CDM power spectrum P(k).

    Uses the BBKS (Bardeen et al. 1986) transfer function with shape
    parameter Γ = Ω_m h ≈ 0.21.

    Parameters
    ----------
    k : array
        Wavenumber in h/Mpc.
    ns : float
        Scalar spectral index.

    Returns
    -------
    Pk : array
        Power spectrum shape (arbitrary normalisation; rescaled by callers).
    """
    Gamma = 0.21
    q = k / Gamma

    # BBKS transfer function (Bardeen, Bond, Kaiser & Szalay 1986)
    T = jnp.log(1.0 + 2.34 * q) / (2.34 * q) * (
        1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    ) ** (-0.25)

    return k ** ns * T ** 2


def generate_gaussian_field(
    key: jax.Array,
    ngrid: int,
    box_size_mpc: float,
) -> jnp.ndarray:
    """Generate a 3-D Gaussian random field with CDM power spectrum.

    The field is normalised so that σ₈ ≈ 0.8 (variance in spheres of
    R = 8 Mpc/h).

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    ngrid : int
        Number of grid cells per side.
    box_size_mpc : float
        Box side length in Mpc.

    Returns
    -------
    delta : array (ngrid, ngrid, ngrid)
        Density contrast field δ in real space (zero mean).
    """
    dk = 2.0 * jnp.pi / box_size_mpc
    kx = jnp.fft.fftfreq(ngrid, d=box_size_mpc / ngrid) * 2 * jnp.pi
    kz = jnp.fft.rfftfreq(ngrid, d=box_size_mpc / ngrid) * 2 * jnp.pi

    kx3d, ky3d, kz3d = jnp.meshgrid(kx, kx, kz, indexing="ij")
    k_mag = jnp.sqrt(kx3d ** 2 + ky3d ** 2 + kz3d ** 2)
    k_mag = jnp.maximum(k_mag, dk)  # avoid k = 0

    Pk_shape = cdm_power_spectrum(k_mag)

    # Normalise so that the field smoothed with a top-hat of R = 8 Mpc/h
    # has variance σ₈² = 0.64.
    sigma8 = 0.8
    V_cell = (box_size_mpc / ngrid) ** 3
    R8 = 8.0  # Mpc/h
    kR = k_mag * R8
    W = 3.0 * (jnp.sin(kR) - kR * jnp.cos(kR)) / jnp.maximum(kR ** 3, 1e-30)
    W = jnp.where(kR < 0.01, 1.0, W)
    var_raw = jnp.sum(Pk_shape * W ** 2) * dk ** 3 / (2 * jnp.pi) ** 3
    norm = sigma8 ** 2 / jnp.maximum(var_raw, 1e-30)
    Pk = Pk_shape * norm

    # Each complex mode is  amplitude * (N_real + i N_imag) with N ~ N(0,1).
    # E[|N_real + i N_imag|²] = 2, so divide by √2 to get <|δ_k|²> = Pk V_cell.
    amplitude = jnp.sqrt(Pk * V_cell / 2.0)

    k1, k2 = jax.random.split(key)
    noise_real = jax.random.normal(k1, shape=k_mag.shape)
    noise_imag = jax.random.normal(k2, shape=k_mag.shape)

    delta_k = amplitude * (noise_real + 1j * noise_imag)
    delta_k = delta_k.at[0, 0, 0].set(0.0 + 0j)  # zero DC mode → zero mean

    return jnp.fft.irfftn(delta_k, s=(ngrid, ngrid, ngrid))


def zeldovich_displacement(
    key: jax.Array,
    ngrid: int,
    box_size_mpc: float,
    growth_factor: float = 1.0,
) -> jnp.ndarray:
    """Zel'dovich-approximation density field via Lagrangian particle displacement.

    Generates a Gaussian random field δ(k), computes the ZA displacement

        Ψ(k) = −i k / k²  δ(k)

    displaces ngrid³ particles from a uniform Lagrangian grid by D(z) Ψ(q),
    and estimates the Eulerian density via NGP mass assignment.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    ngrid : int
        Grid resolution per side.
    box_size_mpc : float
        Box size in Mpc.
    growth_factor : float
        Linear growth factor D(z) (Carroll, Press & Turner 1992) — scales
        the displacement amplitude.

    Returns
    -------
    density : array (ngrid, ngrid, ngrid)
        Eulerian density field 1 + δ_ZA, normalised to mean = 1.
        Voids approach zero; walls/filaments exceed unity.
    """
    delta = generate_gaussian_field(key, ngrid, box_size_mpc)

    # Back to Fourier space to derive the displacement potential.
    delta_k = jnp.fft.rfftn(delta)

    kx = jnp.fft.fftfreq(ngrid, d=box_size_mpc / ngrid) * 2 * jnp.pi
    kz_r = jnp.fft.rfftfreq(ngrid, d=box_size_mpc / ngrid) * 2 * jnp.pi
    kx3d, ky3d, kz3d = jnp.meshgrid(kx, kx, kz_r, indexing="ij")
    k2 = jnp.maximum(kx3d ** 2 + ky3d ** 2 + kz3d ** 2, 1e-30)

    # Zel'dovich displacement: Ψ_i(k) = −i k_i / k² · δ(k)
    psi_x = jnp.fft.irfftn(-1j * kx3d / k2 * delta_k, s=(ngrid, ngrid, ngrid))
    psi_y = jnp.fft.irfftn(-1j * ky3d / k2 * delta_k, s=(ngrid, ngrid, ngrid))
    psi_z = jnp.fft.irfftn(-1j * kz3d / k2 * delta_k, s=(ngrid, ngrid, ngrid))

    # Initial Lagrangian grid: one particle per cell at cell centres.
    cell = box_size_mpc / ngrid
    idx = jnp.arange(ngrid, dtype=jnp.float32)
    qx, qy, qz = jnp.meshgrid(idx, idx, idx, indexing="ij")
    qx_phys = (qx + 0.5) * cell
    qy_phys = (qy + 0.5) * cell
    qz_phys = (qz + 0.5) * cell

    # Displace and apply periodic boundary conditions.
    px = (qx_phys + growth_factor * psi_x) % box_size_mpc
    py = (qy_phys + growth_factor * psi_y) % box_size_mpc
    pz = (qz_phys + growth_factor * psi_z) % box_size_mpc

    # CIC (Cloud-in-Cell) mass assignment.
    # Each particle deposits fractional mass to the 8 surrounding cells based
    # on its sub-cell position.  This gives a continuous density estimate
    # without the integer-count discreteness of NGP and without needing an
    # extra Gaussian smoothing pass that would erase small-scale structure at
    # high z (where displacements are small).  The CIC kernel is a triangular
    # window of width ~1 cell per axis.
    px_np = np.array(px)
    py_np = np.array(py)
    pz_np = np.array(pz)

    # Normalised positions in cell units [0, ngrid)
    xc = (px_np / cell) % ngrid
    yc = (py_np / cell) % ngrid
    zc = (pz_np / cell) % ngrid

    # Lower-left cell index and fractional offset within cell
    ix0 = np.floor(xc).astype(np.int32) % ngrid
    iy0 = np.floor(yc).astype(np.int32) % ngrid
    iz0 = np.floor(zc).astype(np.int32) % ngrid
    ix1 = (ix0 + 1) % ngrid
    iy1 = (iy0 + 1) % ngrid
    iz1 = (iz0 + 1) % ngrid

    dx = (xc - np.floor(xc)).ravel()
    dy = (yc - np.floor(yc)).ravel()
    dz = (zc - np.floor(zc)).ravel()

    density_flat = np.zeros(ngrid ** 3, dtype=np.float64)
    for gx, gy, gz, wx, wy, wz in [
        (ix0, iy0, iz0, 1 - dx, 1 - dy, 1 - dz),
        (ix1, iy0, iz0,     dx, 1 - dy, 1 - dz),
        (ix0, iy1, iz0, 1 - dx,     dy, 1 - dz),
        (ix0, iy0, iz1, 1 - dx, 1 - dy,     dz),
        (ix1, iy1, iz0,     dx,     dy, 1 - dz),
        (ix1, iy0, iz1,     dx, 1 - dy,     dz),
        (ix0, iy1, iz1, 1 - dx,     dy,     dz),
        (ix1, iy1, iz1,     dx,     dy,     dz),
    ]:
        flat = np.ravel_multi_index(
            [gx.ravel(), gy.ravel(), gz.ravel()], (ngrid, ngrid, ngrid)
        )
        density_flat += np.bincount(flat, weights=wx * wy * wz,
                                    minlength=ngrid ** 3)

    density_np = density_flat.reshape(ngrid, ngrid, ngrid).astype(np.float32)

    # Normalise to mean = 1.
    density_np /= density_np.mean()

    return jnp.array(density_np)


def growth_factor_approx(z: float, Om0: float = 0.3) -> float:
    """Approximate linear growth factor D(z) normalised to D(0) = 1.

    Carroll, Press & Turner (1992) fitting formula.
    """
    Ol0 = 1.0 - Om0
    a = 1.0 / (1.0 + z)
    Om_z = Om0 / (Om0 + Ol0 * a ** 3)
    Ol_z = Ol0 * a ** 3 / (Om0 + Ol0 * a ** 3)
    g_z = (5.0 / 2.0) * Om_z / (
        Om_z ** (4.0 / 7.0) - Ol_z + (1.0 + Om_z / 2.0) * (1.0 + Ol_z / 70.0)
    )
    D_z = a * g_z

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
        The density field (1 + δ_ZA).
    x, y, z_pos : arrays
        Comoving positions in Mpc, wrapped into [0, box_size).
    box_size_mpc : float

    Returns
    -------
    density : array
        1 + δ at each position.
    """
    ngrid = density_field.shape[0]
    cell_size = box_size_mpc / ngrid

    ix = ((x % box_size_mpc) / cell_size).astype(int) % ngrid
    iy = ((y % box_size_mpc) / cell_size).astype(int) % ngrid
    iz = ((z_pos % box_size_mpc) / cell_size).astype(int) % ngrid

    return density_field[ix, iy, iz]
