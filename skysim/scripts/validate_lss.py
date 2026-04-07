"""Validate that LSS modulation preserves galaxy number counts and GSMF.

Compares three catalogs generated from the same wide tile:
  - expected: theoretical N(z) and Φ(M) from the Weaver+23 Schechter function
  - flat: Poisson-sampled catalog with no LSS weighting
  - LSS:  same catalog after ZA acceptance-rejection

Key diagnostics:
  1. N(z) — number counts per redshift bin vs. expected
  2. GSMF — stellar mass function shape per redshift bin
  3. Mass-function ratio (LSS / flat) to detect any mass-dependent bias
     (the acceptance-rejection is position-only so the GSMF shape must
     be preserved even if the total count differs)

Usage
-----
    python -m skysim.scripts.validate_lss [--output-dir plots/] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from skysim.config import SimConfig
from skysim.coordinates import TileInfo
from skysim.layers.galaxies import GalaxyLayer
from skysim.layers.lss import growth_factor_approx, zeldovich_displacement
from skysim.models.schechter import (
    double_schechter_phi,
    expected_count_in_volume,
    weaver23_params,
)
from skysim.seed import master_key
from skysim.utils.cosmology import comoving_volume_shell


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------

WIDE_DEG  = 3.0       # field width — enough galaxies per bin for statistics
LSS_BOX   = 300.0     # Mpc — cells ~4.7 Mpc, keeps max_density ~4–6
LSS_NGRID = 64
Z_MAX     = 3.0       # cover multiple redshift bins


def _make_wide_tile(ra_c: float = 180.0, dec_c: float = 0.0) -> TileInfo:
    return TileInfo(
        tile_index=0, nside=64,
        ra_center=ra_c, dec_center=dec_c,
        area_arcmin2=(WIDE_DEG * 60.0) ** 2,
    )


def generate_catalogs(seed: int = 42) -> tuple[dict, dict, object]:
    """Return (cat_flat, cat_lss, density_field)."""
    config = SimConfig(seed=seed, nside=64, z_max=Z_MAX, n_redshift_bins=60)
    tile = _make_wide_tile()
    layer = GalaxyLayer()

    base_key = master_key(seed)
    k_flat, k_lss = jax.random.split(base_key)

    print("  Generating flat catalog …")
    cat_flat = layer.generate_catalog(k_flat, tile, config, density_field=None)

    lss_key = jax.random.PRNGKey(seed + 99)
    gf = float(growth_factor_approx(0.5))
    density_field = zeldovich_displacement(lss_key, LSS_NGRID, LSS_BOX,
                                           growth_factor=gf)

    print(f"  LSS density field: mean={float(density_field.mean()):.3f}  "
          f"max={float(density_field.max()):.2f}  "
          f"std={float(density_field.std()):.3f}")
    print(f"  Effective acceptance rate with oversample=2: "
          f"{2.0 / float(density_field.max()):.1%}  "
          f"(target ~100%)")

    print("  Generating LSS catalog …")
    cat_lss = layer.generate_catalog(
        k_lss, tile, config,
        density_field=density_field, density_box_mpc=LSS_BOX,
    )

    return (
        {k: np.array(v) for k, v in cat_flat.items()},
        {k: np.array(v) for k, v in cat_lss.items()},
        density_field,
    )


# ---------------------------------------------------------------------------
# Expected counts from Schechter model
# ---------------------------------------------------------------------------

Z_BINS = [(0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
AREA_ARCMIN2 = (WIDE_DEG * 60.0) ** 2
LOG_M_MIN = 7.0


def expected_n_per_bin() -> list[float]:
    out = []
    for z_lo, z_hi in Z_BINS:
        z_mid = 0.5 * (z_lo + z_hi)
        vol = comoving_volume_shell(z_lo, z_hi, AREA_ARCMIN2)
        params = weaver23_params(z_mid)
        out.append(expected_count_in_volume(params, vol, log_m_min=LOG_M_MIN))
    return out


# ---------------------------------------------------------------------------
# Plot 1: N(z) — total counts per bin
# ---------------------------------------------------------------------------

def plot_nz(cat_flat: dict, cat_lss: dict, output_dir: Path) -> None:
    z_flat = cat_flat["z"]
    z_lss  = cat_lss["z"]
    expected = expected_n_per_bin()

    bin_labels  = [f"z={lo}–{hi}" for lo, hi in Z_BINS]
    n_expected  = np.array(expected)
    n_flat      = np.array([((z_flat >= lo) & (z_flat < hi)).sum()
                            for lo, hi in Z_BINS], dtype=float)
    n_lss       = np.array([((z_lss  >= lo) & (z_lss  < hi)).sum()
                            for lo, hi in Z_BINS], dtype=float)

    x = np.arange(len(Z_BINS))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Absolute counts ---
    ax = axes[0]
    ax.bar(x - w,   n_expected, width=w, label="Schechter (expected)",
           color="grey",     alpha=0.7, edgecolor="k", linewidth=0.6)
    ax.bar(x,       n_flat,     width=w, label="Flat (no LSS)",
           color="steelblue", alpha=0.8, edgecolor="k", linewidth=0.6)
    ax.bar(x + w,   n_lss,      width=w, label="With LSS",
           color="tomato",    alpha=0.8, edgecolor="k", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=15)
    ax.set_ylabel("Galaxy count")
    ax.set_title("N(z): absolute counts")
    ax.legend()
    ax.set_yscale("log")
    for xi, ne, nf, nl in zip(x, n_expected, n_flat, n_lss):
        if ne > 0:
            ax.annotate(f"{nf/ne:.2f}", xy=(xi,       nf), ha="center",
                        va="bottom", fontsize=7, color="steelblue")
            ax.annotate(f"{nl/ne:.2f}", xy=(xi + w,   nl), ha="center",
                        va="bottom", fontsize=7, color="tomato")

    # --- Ratio to expected ---
    ax2 = axes[1]
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ratio_flat = np.where(n_expected > 0, n_flat / n_expected, np.nan)
    ratio_lss  = np.where(n_expected > 0, n_lss  / n_expected, np.nan)
    ax2.plot(x, ratio_flat, "o-", color="steelblue", linewidth=1.5,
             label="Flat / expected")
    ax2.plot(x, ratio_lss,  "s-", color="tomato",    linewidth=1.5,
             label="LSS / expected")
    ax2.fill_between(x, 0.9, 1.1, alpha=0.1, color="grey", label="±10%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=15)
    ax2.set_ylabel("N / N_expected")
    ax2.set_title("N(z): ratio to Schechter model")
    ax2.legend()
    ax2.set_ylim(0, 1.6)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Galaxy number counts — {WIDE_DEG}°×{WIDE_DEG}° field\n"
        f"flat N={len(z_flat):,}   LSS N={len(z_lss):,}   "
        f"expected N={int(sum(n_expected)):,}",
        fontsize=11,
    )
    fig.tight_layout()
    out = output_dir / "lss_nz_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: GSMF — shape and normalisation per redshift bin
# ---------------------------------------------------------------------------

def plot_gsmf(cat_flat: dict, cat_lss: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    all_axes = axes.flat

    bins = np.linspace(7.0, 12.0, 25)
    bin_c = 0.5 * (bins[:-1] + bins[1:])
    dlogm = bins[1] - bins[0]

    for ax, (z_lo, z_hi) in zip(all_axes, Z_BINS):
        m_flat = cat_flat["log_mass"][(cat_flat["z"] >= z_lo) & (cat_flat["z"] < z_hi)]
        m_lss  = cat_lss["log_mass"][ (cat_lss["z"]  >= z_lo) & (cat_lss["z"]  < z_hi)]

        vol = comoving_volume_shell(z_lo, z_hi, AREA_ARCMIN2)

        # Absolute Φ(M) in Mpc^-3 dex^-1
        phi_flat, _ = np.histogram(m_flat, bins=bins)
        phi_lss,  _ = np.histogram(m_lss,  bins=bins)
        phi_flat = phi_flat / (dlogm * vol)
        phi_lss  = phi_lss  / (dlogm * vol)

        # Theoretical Weaver+23 Schechter function
        z_mid = 0.5 * (z_lo + z_hi)
        params = weaver23_params(z_mid)
        phi_th = np.array(double_schechter_phi(jnp.array(bin_c), params))

        # Poisson error bars
        n_flat_raw, _ = np.histogram(m_flat, bins=bins)
        n_lss_raw,  _ = np.histogram(m_lss,  bins=bins)
        err_flat = np.sqrt(n_flat_raw) / (dlogm * vol)
        err_lss  = np.sqrt(n_lss_raw)  / (dlogm * vol)

        ax.errorbar(bin_c, phi_flat, yerr=err_flat, fmt="o-", ms=3,
                    color="steelblue", linewidth=1.2, label=f"Flat (N={len(m_flat):,})")
        ax.errorbar(bin_c, phi_lss,  yerr=err_lss,  fmt="s-", ms=3,
                    color="tomato",    linewidth=1.2, label=f"LSS  (N={len(m_lss):,})")
        ax.plot(bin_c, phi_th, "k--", linewidth=1.5, label="Weaver+23")

        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e-1)
        ax.set_xlim(7.5, 12.0)
        ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        ax.set_ylabel(r"$\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$")
        ax.set_title(f"z = {z_lo}–{z_hi}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, which="both")

    # Use remaining (6th) panel for ratio summary
    axes_list = list(axes.flat)
    ax_r = axes_list[5]
    ax_r.set_visible(True)
    if len(Z_BINS) == 5:
        ax_r = list(axes.flat)[5]
        ax_r.set_visible(True)
        for (z_lo, z_hi), col in zip(Z_BINS,
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]):
            m_flat = cat_flat["log_mass"][
                (cat_flat["z"] >= z_lo) & (cat_flat["z"] < z_hi)]
            m_lss  = cat_lss["log_mass"][
                (cat_lss["z"]  >= z_lo) & (cat_lss["z"]  < z_hi)]
            phi_flat, _ = np.histogram(m_flat, bins=bins)
            phi_lss,  _ = np.histogram(m_lss,  bins=bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(phi_flat > 0, phi_lss / phi_flat, np.nan)
            ax_r.plot(bin_c, ratio, "-o", ms=3, color=col,
                      label=f"z={z_lo}–{z_hi}")
        ax_r.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
        ax_r.fill_between(bin_c, 0.9, 1.1, alpha=0.1, color="grey")
        ax_r.set_ylim(0, 2.0)
        ax_r.set_xlim(7.5, 12.0)
        ax_r.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        ax_r.set_ylabel("LSS / flat")
        ax_r.set_title("GSMF ratio (LSS / flat)\n(flat → mass-independent if 1)")
        ax_r.legend(fontsize=7)
        ax_r.grid(True, alpha=0.2)

    fig.suptitle(
        f"Galaxy Stellar Mass Function — {WIDE_DEG}°×{WIDE_DEG}° field",
        fontsize=13,
    )
    fig.tight_layout()
    out = output_dir / "lss_gsmf_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate LSS number counts / GSMF")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cat_flat, cat_lss, density_field = generate_catalogs(seed=args.seed)

    print(f"  Flat:  {len(cat_flat['z']):,} galaxies")
    print(f"  LSS:   {len(cat_lss['z']):,} galaxies")
    print(f"  Expected total: {int(sum(expected_n_per_bin())):,} galaxies")

    plot_nz(cat_flat, cat_lss, args.output_dir)
    plot_gsmf(cat_flat, cat_lss, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
