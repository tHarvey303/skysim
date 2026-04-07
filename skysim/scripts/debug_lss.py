"""Debug plot: visualise the large-scale structure density field.

Shows voids and overdensities produced by the Zel'dovich-approximation
Gaussian random field with CDM power spectrum, and demonstrates how the
field modulates galaxy surface density.

Usage
-----
    python -m skysim.scripts.debug_lss [--seed 42] [--ngrid 64]
    [--box 300] [--output-dir plots/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec

from skysim.layers.lss import (
    generate_gaussian_field,
    growth_factor_approx,
    zeldovich_displacement,
)
from skysim.utils.cosmology import COSMO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comoving_to_z(dc_mpc: np.ndarray) -> np.ndarray:
    """Approximate redshift from comoving distance (Mpc)."""
    z_table = np.linspace(0.0, 6.0, 2000)
    dc_table = COSMO.comoving_distance(z_table).value
    return np.interp(dc_mpc, dc_table, z_table)


def _project_along_axis(field: np.ndarray, axis: int) -> np.ndarray:
    """Mean-project a 3-D field along one axis."""
    return field.mean(axis=axis)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_lss_debug_plots(
    seed: int = 42,
    ngrid: int = 64,
    box_mpc: float = 300.0,
    output_dir: Path = Path("plots"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(seed)

    # --- Generate density fields at four epochs from the SAME initial conditions ---
    # All redshifts use the same PRNG key so we see the same structure displaced
    # by different amounts — only the growth factor D(z) changes.
    z_vals = [0.0, 0.5, 1.0, 2.0]
    fields = {}
    for z in z_vals:
        gf = float(growth_factor_approx(z))
        fields[z] = np.array(zeldovich_displacement(key, ngrid, box_mpc, growth_factor=gf))

    field_z0 = fields[0.0]

    # -----------------------------------------------------------------------
    # Figure 1: 3 orthogonal slices of the z=0 density field
    # -----------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle(
        f"LSS density field (1+δ) slices — z=0  |  N={ngrid}³, L={box_mpc} Mpc/h  |  seed={seed}",
        fontsize=13,
    )

    mid = ngrid // 2
    slice_data = [
        (field_z0[:, :, mid], "XY (z-slice)", "x [Mpc/h]", "y [Mpc/h]"),
        (field_z0[:, mid, :], "XZ (y-slice)", "x [Mpc/h]", "z [Mpc/h]"),
        (field_z0[mid, :, :], "YZ (x-slice)", "y [Mpc/h]", "z [Mpc/h]"),
    ]

    vmin = max(0.0, field_z0.mean() - 2.5 * field_z0.std())
    vmax = field_z0.mean() + 2.5 * field_z0.std()

    for ax, (data, title, xlabel, ylabel) in zip(axes1, slice_data):
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin, vmax=vmax,
            extent=[0, box_mpc, 0, box_mpc],
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig1.colorbar(im, ax=ax, label="1+δ", fraction=0.046, pad=0.04)

    fig1.tight_layout()
    out1 = output_dir / "lss_slices.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {out1}")

    # -----------------------------------------------------------------------
    # Figure 2: Line-of-sight projection at four redshifts
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
    fig2.suptitle(
        "LSS projected density field (1+δ) at four epochs",
        fontsize=13,
    )

    for col, z in enumerate(z_vals):
        f = fields[z]
        proj = _project_along_axis(f, axis=2)  # project along LOS axis

        # Raw projected density
        ax_top = axes2[0, col]
        im_t = ax_top.imshow(
            proj.T,
            origin="lower",
            cmap="inferno",
            extent=[0, box_mpc, 0, box_mpc],
            interpolation="bilinear",
        )
        ax_top.set_title(f"z = {z:.1f}  (D(z)={growth_factor_approx(z):.2f})")
        ax_top.set_xlabel("x [Mpc/h]")
        if col == 0:
            ax_top.set_ylabel("y [Mpc/h]")
        fig2.colorbar(im_t, ax=ax_top, label="⟨1+δ⟩", fraction=0.046, pad=0.04)

        # Density contrast δ = (1+δ) - 1
        ax_bot = axes2[1, col]
        delta = proj - 1.0
        lim = max(abs(delta.min()), abs(delta.max()))
        im_b = ax_bot.imshow(
            delta.T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-lim, vmax=lim,
            extent=[0, box_mpc, 0, box_mpc],
            interpolation="bilinear",
        )
        ax_bot.set_xlabel("x [Mpc/h]")
        if col == 0:
            ax_bot.set_ylabel("y [Mpc/h]")
        fig2.colorbar(im_b, ax=ax_bot, label="⟨δ⟩", fraction=0.046, pad=0.04)

    fig2.tight_layout()
    out2 = output_dir / "lss_redshift_evolution.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # -----------------------------------------------------------------------
    # Figure 3: Galaxy density modulation — with vs without LSS
    # -----------------------------------------------------------------------
    _make_galaxy_comparison_plot(seed, ngrid, box_mpc, output_dir)

    # -----------------------------------------------------------------------
    # Figure 4: Power spectrum of the density field
    # -----------------------------------------------------------------------
    _make_power_spectrum_plot(fields, box_mpc, ngrid, output_dir)

    # -----------------------------------------------------------------------
    # Figure 5: PDF of density contrast at different redshifts
    # -----------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    bins = np.linspace(-3, 6, 80)

    for z in z_vals:
        delta_vals = (fields[z] - 1.0).ravel()
        ax5.hist(
            delta_vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"z = {z:.1f}",
        )

    ax5.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax5.set_xlabel("Density contrast δ", fontsize=12)
    ax5.set_ylabel("Probability density", fontsize=12)
    ax5.set_title("PDF of δ at different redshifts", fontsize=13)
    ax5.legend()
    ax5.set_xlim(bins[0], bins[-1])
    fig5.tight_layout()
    out5 = output_dir / "lss_density_pdf.png"
    fig5.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"Saved: {out5}")

    print(f"\nAll LSS debug plots saved to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Helper: galaxy surface density comparison
# ---------------------------------------------------------------------------

def _make_galaxy_comparison_plot(
    seed: int, ngrid: int, box_mpc: float, output_dir: Path
) -> None:
    """Wedge diagram showing LSS modulation.

    Uses a wide custom tile (8°×8°) and a box large enough to cover the
    shown redshift range without wrapping, so voids and overdensities are
    clearly visible in both the density-field background and galaxy counts.
    """
    from scipy.ndimage import gaussian_filter
    from skysim.config import SimConfig
    from skysim.coordinates import TileInfo
    from skysim.layers.galaxies import GalaxyLayer
    from skysim.seed import layer_key, master_key, tile_key
    from skysim.utils.cosmology import COSMO

    # ---- Parameters chosen for visual clarity --------------------------------
    # Wide tile gives ~10× more galaxies than the default ~1° tile.
    WIDE_DEG = 8.0
    # Box large enough that z_max ~ 0.3 fits in one period without wrapping.
    # Cells: VIS_BOX / VIS_NGRID ≈ 1200/64 ≈ 19 Mpc — fine enough for
    # interesting structure, coarse enough to create clear void/wall features.
    VIS_BOX = 1200.0   # Mpc/h
    VIS_NGRID = 64
    DC_MAX = VIS_BOX   # show exactly one box period along LOS
    # --------------------------------------------------------------------------

    ra_c, dec_c = 180.0, 0.0
    wide_tile = TileInfo(
        tile_index=0, nside=64, ra_center=ra_c, dec_center=dec_c,
        area_arcmin2=(WIDE_DEG * 60.0) ** 2,
    )

    # Restrict z_max so the catalog only covers one box period (DC_MAX Mpc).
    z_table = np.linspace(0.0, 2.0, 5000)
    dc_table = COSMO.comoving_distance(z_table).value
    z_max_vis = float(np.interp(DC_MAX, dc_table, z_table))
    config = SimConfig(seed=seed, nside=64, z_max=z_max_vis, n_redshift_bins=30)

    layer = GalaxyLayer()
    base_key = master_key(seed)
    k_no_lss, k_lss = jax.random.split(base_key)

    # Density field sized for this visualisation
    lss_key = jax.random.PRNGKey(seed + 1)
    gf = float(growth_factor_approx(0.15))   # growth factor at survey midpoint
    density_field = zeldovich_displacement(lss_key, VIS_NGRID, VIS_BOX, growth_factor=gf)

    print("  Generating flat catalog …")
    cat_flat = layer.generate_catalog(k_no_lss, wide_tile, config, density_field=None)
    print("  Generating LSS catalog …")
    cat_lss = layer.generate_catalog(
        k_lss, wide_tile, config,
        density_field=density_field, density_box_mpc=VIS_BOX,
    )

    # Comoving distances
    dc_flat = COSMO.comoving_distance(np.array(cat_flat["z"])).value
    dc_lss  = COSMO.comoving_distance(np.array(cat_lss["z"])).value
    ra_flat = np.array(cat_flat["ra"])
    ra_lss  = np.array(cat_lss["ra"])

    # Clip to one box period (no wrapping artifacts)
    m_flat = dc_flat < DC_MAX
    m_lss  = dc_lss  < DC_MAX

    # ---- Density-field image on the (RA, dc) wedge plane --------------------
    # Use the Dec-centre slice of the box (iy = VIS_NGRID//2) — averaging over
    # the Dec axis kills the contrast (mean of a zero-mean field → 0).
    n_ra_img, n_dc_img = 400, 400
    ra_img = np.linspace(ra_c - WIDE_DEG / 2, ra_c + WIDE_DEG / 2, n_ra_img)
    dc_img = np.linspace(5.0, DC_MAX, n_dc_img)
    RA_g, DC_g = np.meshgrid(ra_img, dc_img, indexing="ij")
    dra_rad = (RA_g - ra_c) * (np.pi / 180.0)
    x_mpc = DC_g * dra_rad
    z_mpc = DC_g
    cell = VIS_BOX / VIS_NGRID
    ix = ((x_mpc % VIS_BOX) / cell).astype(int) % VIS_NGRID
    iz = ((z_mpc % VIS_BOX) / cell).astype(int) % VIS_NGRID
    iy_mid = VIS_NGRID // 2
    field_arr = np.array(density_field)
    field_wedge = field_arr[ix, iy_mid, iz]          # single Dec slice
    field_wedge = gaussian_filter(field_wedge, sigma=1.2)
    delta_wedge = field_wedge - 1.0
    clim = np.percentile(np.abs(delta_wedge), 97)    # symmetric colour limits

    # ---- Galaxy density histograms -------------------------------------------
    # Use coarser bins aligned with the cell scale (~19 Mpc) so individual cells
    # are visible; smooth lightly to reduce shot noise.
    n_dc_bins = int(DC_MAX / (VIS_BOX / VIS_NGRID)) + 1  # ≈ one bin per cell
    n_ra_bins = max(30, int(WIDE_DEG * 4))
    dc_bins = np.linspace(0, DC_MAX, n_dc_bins)
    ra_bins = np.linspace(ra_c - WIDE_DEG / 2, ra_c + WIDE_DEG / 2, n_ra_bins)

    def _make_hist(ra, dc, mask):
        h, _, _ = np.histogram2d(dc[mask], ra[mask], bins=[dc_bins, ra_bins])
        return gaussian_filter(h.T.astype(float), sigma=1.0)  # (ra, dc)

    h_flat_s = _make_hist(ra_flat, dc_flat, m_flat)
    h_lss_s  = _make_hist(ra_lss,  dc_lss,  m_lss)

    # Ratio: divide column-by-column so the selection function cancels out.
    # Each column is a dc shell; normalise LSS counts by flat counts.
    col_flat = h_flat_s.sum(axis=0, keepdims=True)  # (1, n_dc)
    col_lss  = h_lss_s.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_2d = np.where(col_flat > 0, h_lss_s / (h_flat_s + 1e-6), np.nan)
        # Normalise so mean ratio per dc-shell = 1 (removes radial trend)
        shell_mean = np.nanmean(ratio_2d, axis=0, keepdims=True)
        ratio_2d = np.where(np.isfinite(ratio_2d),
                            ratio_2d / np.where(shell_mean > 0, shell_mean, 1.0),
                            np.nan)
    ratio_2d = gaussian_filter(np.nan_to_num(ratio_2d, nan=1.0), sigma=1.0)

    # ---- Figure: 4 panels ----------------------------------------------------
    fig = plt.figure(figsize=(22, 5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.38)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    extent_wd   = [0, DC_MAX, ra_c - WIDE_DEG / 2, ra_c + WIDE_DEG / 2]
    extent_hist = [dc_bins[0], dc_bins[-1], ra_bins[0], ra_bins[-1]]

    # Panel 0: density-field slice
    im0 = axes[0].imshow(
        delta_wedge, origin="lower", cmap="RdBu_r",
        vmin=-clim, vmax=clim, aspect="auto", extent=extent_wd,
        interpolation="bilinear",
    )
    axes[0].set_title("Density field δ\n(Dec-centre slice of box)")
    axes[0].set_xlabel("Comoving distance [Mpc]")
    axes[0].set_ylabel("RA [deg]")
    fig.colorbar(im0, ax=axes[0], label="δ", fraction=0.046, pad=0.04)
    _add_redshift_ticks(axes[0], dc_img, COSMO)

    # Panel 1: flat galaxy density (raw, for reference)
    vmax_gal = np.percentile(h_flat_s, 99.5)
    kw_gal = dict(origin="lower", cmap="hot_r", vmin=0, vmax=vmax_gal,
                  aspect="auto", interpolation="bilinear", extent=extent_hist)
    im1 = axes[1].imshow(h_flat_s, **kw_gal)
    axes[1].set_title(f"No LSS  (N={m_flat.sum():,})\ngalaxy density")
    axes[1].set_xlabel("Comoving distance [Mpc]")
    axes[1].set_ylabel("RA [deg]")
    fig.colorbar(im1, ax=axes[1], label="count / bin", fraction=0.046, pad=0.04)
    _add_redshift_ticks(axes[1], dc_bins, COSMO)

    # Panel 2: LSS/flat ratio — selection function cancels, pure density signal
    rlim = np.percentile(np.abs(ratio_2d - 1.0), 97) + 1.0
    im2 = axes[2].imshow(
        ratio_2d, origin="lower", cmap="RdBu_r",
        vmin=2.0 - rlim, vmax=rlim, aspect="auto",
        interpolation="bilinear", extent=extent_hist,
    )
    axes[2].set_title(f"LSS / flat ratio  (N_LSS={m_lss.sum():,})\nnormalised per dc shell")
    axes[2].set_xlabel("Comoving distance [Mpc]")
    axes[2].set_ylabel("RA [deg]")
    fig.colorbar(im2, ax=axes[2], label="relative density", fraction=0.046, pad=0.04)
    _add_redshift_ticks(axes[2], dc_bins, COSMO)

    # Panel 3: 1-D LOS ratio — shows voids/walls as deviations from unity
    ax3 = axes[3]
    n_flat_1d = np.maximum(np.histogram(dc_flat[m_flat], bins=dc_bins)[0], 1).astype(float)
    n_lss_1d  = np.histogram(dc_lss[m_lss],  bins=dc_bins)[0].astype(float)
    dc_c = 0.5 * (dc_bins[:-1] + dc_bins[1:])
    ratio_1d = n_lss_1d / n_flat_1d * (m_flat.sum() / max(m_lss.sum(), 1))
    ratio_1d_smooth = gaussian_filter(ratio_1d, sigma=1.5)
    ax3.axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    ax3.plot(dc_c, ratio_1d_smooth, color="purple", linewidth=1.6,
             label="LSS / flat (smoothed)")
    ax3.fill_between(dc_c, ratio_1d_smooth, 1.0,
                     where=ratio_1d_smooth < 1.0, alpha=0.3, color="navy",
                     label="voids")
    ax3.fill_between(dc_c, ratio_1d_smooth, 1.0,
                     where=ratio_1d_smooth > 1.0, alpha=0.3, color="crimson",
                     label="walls / filaments")
    ax3.set_xlabel("Comoving distance [Mpc]")
    ax3.set_ylabel("N_LSS / N_flat  (normalised)")
    ax3.set_title("LOS density ratio\n(voids < 1 < walls)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    _add_redshift_ticks(ax3, dc_bins, COSMO, which="top")

    fig.suptitle(
        f"LSS effect on galaxy distribution — 8°×8° field, one box period "
        f"({VIS_BOX:.0f} Mpc/h, {VIS_NGRID}³ grid, seed={seed})",
        fontsize=12, y=1.01,
    )
    out = output_dir / "lss_galaxy_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _add_redshift_ticks(ax, dc_bins, cosmo, which: str = "bottom") -> None:
    """Add a secondary redshift axis tick overlay on a comoving-distance axis."""
    z_ticks = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    dc_ticks = [cosmo.comoving_distance(z).value for z in z_ticks]
    dc_min, dc_max = dc_bins[0], dc_bins[-1]
    dc_ticks_valid = [(dc, z) for dc, z in zip(dc_ticks, z_ticks)
                      if dc_min <= dc <= dc_max]
    if not dc_ticks_valid:
        return
    ax2 = ax.twiny() if which == "top" else ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([dc for dc, _ in dc_ticks_valid])
    ax2.set_xticklabels([f"z={z}" for _, z in dc_ticks_valid], fontsize=7)
    ax2.tick_params(axis="x", direction="in", length=3)


# ---------------------------------------------------------------------------
# Helper: power spectrum
# ---------------------------------------------------------------------------

def _make_power_spectrum_plot(
    fields: dict, box_mpc: float, ngrid: int, output_dir: Path
) -> None:
    """Measure and plot the 1-D power spectrum of the density fields."""
    from skysim.layers.lss import cdm_power_spectrum

    fig, ax = plt.subplots(figsize=(7, 5))

    dk = 2.0 * np.pi / box_mpc
    kx = np.fft.fftfreq(ngrid, d=box_mpc / ngrid) * 2 * np.pi
    kz = np.fft.rfftfreq(ngrid, d=box_mpc / ngrid) * 2 * np.pi
    kx3, ky3, kz3 = np.meshgrid(kx, kx, kz, indexing="ij")
    k_mag = np.sqrt(kx3**2 + ky3**2 + kz3**2)

    k_bins = np.logspace(np.log10(dk * 1.1), np.log10(np.pi * ngrid / box_mpc), 30)
    k_centres = 0.5 * (k_bins[:-1] + k_bins[1:])

    for z, f in fields.items():
        delta_k = np.fft.rfftn(f - 1.0)
        power_k = (np.abs(delta_k) ** 2) * (box_mpc / ngrid) ** 3

        Pk_bins = []
        for klo, khi in zip(k_bins[:-1], k_bins[1:]):
            mask = (k_mag >= klo) & (k_mag < khi)
            Pk_bins.append(power_k[mask].mean() if mask.sum() > 0 else np.nan)

        Pk_bins = np.array(Pk_bins)
        ax.loglog(k_centres, k_centres**3 * Pk_bins / (2 * np.pi**2),
                  label=f"z = {z:.1f}", linewidth=1.8)

    # Theoretical CDM shape — measure z=0 spectrum and normalise to it
    z0_delta_k = np.fft.rfftn(fields[0.0] - 1.0)
    z0_power = (np.abs(z0_delta_k) ** 2) * (box_mpc / ngrid) ** 3
    z0_Pk = []
    for klo, khi in zip(k_bins[:-1], k_bins[1:]):
        mask = (k_mag >= klo) & (k_mag < khi)
        z0_Pk.append(z0_power[mask].mean() if mask.sum() > 0 else np.nan)
    z0_Pk = np.array(z0_Pk)

    Pk_th_norm = np.array(cdm_power_spectrum(jnp.array(k_centres)))
    delta2_z0 = k_centres ** 3 * z0_Pk / (2 * np.pi ** 2)
    delta2_th = k_centres ** 3 * Pk_th_norm / (2 * np.pi ** 2)
    # Scale theory curve to match z=0 measurement at peak
    valid = np.isfinite(delta2_z0) & np.isfinite(delta2_th) & (delta2_th > 0)
    if valid.any():
        scale = np.nanmax(delta2_z0[valid]) / np.nanmax(delta2_th[valid])
    else:
        scale = 1.0
    ax.loglog(k_centres, delta2_th * scale,
              "k--", linewidth=1.2, alpha=0.6, label="CDM shape (theory)")

    ax.set_xlabel("k  [h/Mpc]", fontsize=12)
    ax.set_ylabel(r"$\Delta^2(k) = k^3 P(k) / 2\pi^2$", fontsize=12)
    ax.set_title("Dimensionless power spectrum", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = output_dir / "lss_power_spectrum.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSS debug plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngrid", type=int, default=64,
                        help="Grid resolution per side (default 64)")
    parser.add_argument("--box", type=float, default=300.0,
                        help="Box size in Mpc/h (default 300)")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"),
                        help="Output directory for plots")
    args = parser.parse_args()

    make_lss_debug_plots(
        seed=args.seed,
        ngrid=args.ngrid,
        box_mpc=args.box,
        output_dir=args.output_dir,
    )
