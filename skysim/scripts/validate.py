"""Validation script: compare generated catalog distributions to input models.

Generates a galaxy catalog for a representative tile and produces diagnostic
plots comparing the sampled distributions (mass function, mass-size relation,
mass-metallicity relation, magnitude distribution) to the input models.

Usage:
    python -m skysim.scripts.validate [--output-dir plots/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from skysim.config import SimConfig
from skysim.coordinates import TileInfo, radec_to_tile
from skysim.layers.galaxies import GalaxyLayer
from skysim.models.mass_metallicity import oh12_to_mass_fraction, zahid14_metallicity
from skysim.models.mass_size import log_re_mean
from skysim.models.schechter import (
    SchechterParams,
    double_schechter_phi,
    weaver23_params,
)
from skysim.seed import layer_key, master_key, tile_key


def generate_catalog(seed: int = 42, nside: int = 64) -> dict:
    """Generate a galaxy catalog for a single tile."""
    config = SimConfig(seed=seed, nside=nside)
    ra, dec = 180.0, 0.0
    tile_idx = radec_to_tile(nside, ra, dec)
    tile = TileInfo.from_index(nside, tile_idx)

    key = layer_key(tile_key(master_key(seed), tile_idx), "galaxies")
    layer = GalaxyLayer()
    cat = layer.generate_catalog(key, tile, config)

    return {k: np.array(v) for k, v in cat.items()}


def plot_mass_function(cat: dict, output_dir: Path):
    """Plot sampled mass function vs Weaver+23 model in redshift bins."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    z_bins = [(0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0)]

    z = cat["z"]
    log_mass = cat["log_mass"]

    for ax, (z_lo, z_hi) in zip(axes.flat, z_bins):
        mask = (z >= z_lo) & (z < z_hi)
        if mask.sum() < 10:
            ax.set_title(f"z=[{z_lo}, {z_hi})\nN={mask.sum()}")
            continue

        masses = log_mass[mask]
        bins = np.linspace(7, 12, 30)
        counts, edges = np.histogram(masses, bins=bins)

        # Normalize to approximate phi (counts / dlog_m / volume)
        # Just show shape comparison
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        dlogm = edges[1] - edges[0]
        density = counts / (dlogm * mask.sum())

        ax.bar(bin_centers, density, width=dlogm * 0.8, alpha=0.6, label="Sampled")

        # Model
        z_mid = 0.5 * (z_lo + z_hi)
        params = weaver23_params(z_mid)
        log_m_grid = np.linspace(7, 12, 200)
        phi = np.array(double_schechter_phi(jnp.array(log_m_grid), params))
        phi_norm = phi / np.trapz(phi, log_m_grid)
        ax.plot(log_m_grid, phi_norm, "r-", lw=2, label="Weaver+23")

        ax.set_yscale("log")
        ax.set_ylim(1e-4, None)
        ax.set_xlabel(r"$\log_{10}(M/M_\odot)$")
        ax.set_ylabel("Normalized density")
        ax.set_title(f"z=[{z_lo}, {z_hi})  N={mask.sum()}")
        ax.legend(fontsize=8)

    fig.suptitle("Galaxy Stellar Mass Function", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "mass_function.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'mass_function.png'}")


def plot_mass_size(cat: dict, output_dir: Path):
    """Plot mass-size relation vs van der Wel+14."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    log_mass = cat["log_mass"]
    log_re = cat["log_re_kpc"]
    z = cat["z"]

    for ax, (z_lo, z_hi, label) in zip(
        axes, [(0.2, 1.5, "z=0.2-1.5"), (1.5, 4.0, "z=1.5-4.0")]
    ):
        mask = (z >= z_lo) & (z < z_hi)
        if mask.sum() < 10:
            continue

        ax.scatter(
            log_mass[mask], log_re[mask],
            s=1, alpha=0.2, c="steelblue", rasterized=True,
        )

        # Model lines
        z_mid = 0.5 * (z_lo + z_hi)
        m_grid = jnp.linspace(8, 12, 100)
        z_arr = jnp.full(100, z_mid)

        for is_late, color, lbl in [(True, "blue", "Late"), (False, "red", "Early")]:
            re_model = np.array(log_re_mean(
                m_grid, z_arr, jnp.full(100, is_late),
            ))
            ax.plot(m_grid, re_model, color=color, lw=2, label=f"{lbl} (vdW+14)")

        ax.set_xlabel(r"$\log_{10}(M/M_\odot)$")
        ax.set_ylabel(r"$\log_{10}(R_e/\mathrm{kpc})$")
        ax.set_title(f"{label}  N={mask.sum()}")
        ax.legend(fontsize=8)
        ax.set_xlim(7.5, 12)
        ax.set_ylim(-1.5, 2)

    fig.suptitle("Mass-Size Relation", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "mass_size.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'mass_size.png'}")


def plot_mzr(cat: dict, output_dir: Path):
    """Plot mass-metallicity relation vs Zahid+14."""
    fig, ax = plt.subplots(figsize=(8, 5))

    log_mass = cat["log_mass"]
    metallicity = cat["metallicity"]
    z = cat["z"]

    mask = (z > 0.2) & (z < 2.0)
    if mask.sum() > 0:
        ax.scatter(
            log_mass[mask], np.log10(metallicity[mask]),
            s=1, alpha=0.2, c="steelblue", rasterized=True,
        )

    # Model
    m_grid = jnp.linspace(8, 12, 100)
    for z_val, color in [(0.5, "blue"), (1.0, "green"), (2.0, "red")]:
        z_arr = jnp.full(100, z_val)
        oh12 = zahid14_metallicity(m_grid, z_arr)
        Z_model = np.array(oh12_to_mass_fraction(oh12))
        ax.plot(m_grid, np.log10(Z_model), color=color, lw=2, label=f"Zahid+14 z={z_val}")

    ax.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(Z)$")
    ax.set_title("Mass-Metallicity Relation")
    ax.legend(fontsize=9)
    ax.set_xlim(7.5, 12)

    plt.tight_layout()
    fig.savefig(output_dir / "mzr.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'mzr.png'}")


def plot_magnitude_distribution(cat: dict, output_dir: Path):
    """Plot apparent magnitude distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    mag = cat["mag"]
    valid = np.isfinite(mag) & (mag < 35)

    bins = np.arange(16, 32, 0.5)
    ax.hist(mag[valid], bins=bins, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Apparent Magnitude (AB)")
    ax.set_ylabel("Count")
    ax.set_title(f"Magnitude Distribution  (N={valid.sum()})")
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(output_dir / "magnitude_dist.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'magnitude_dist.png'}")


def plot_redshift_distribution(cat: dict, output_dir: Path):
    """Plot redshift distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    z = cat["z"]
    bins = np.linspace(0, 6, 60)
    ax.hist(z, bins=bins, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Count")
    ax.set_title(f"Redshift Distribution  (N={len(z)})")

    plt.tight_layout()
    fig.savefig(output_dir / "redshift_dist.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'redshift_dist.png'}")


def main():
    parser = argparse.ArgumentParser(description="Validate SkySim galaxy populations")
    parser.add_argument("--output-dir", type=str, default="plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nside", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating catalog (seed={args.seed}, nside={args.nside})...")
    cat = generate_catalog(seed=args.seed, nside=args.nside)
    print(f"  Generated {len(cat.get('z', []))} galaxies")

    print("Generating validation plots...")
    plot_mass_function(cat, output_dir)
    plot_mass_size(cat, output_dir)
    plot_mzr(cat, output_dir)
    plot_magnitude_distribution(cat, output_dir)
    plot_redshift_distribution(cat, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
