"""Image renderer: assembles layers into a final telescope image.

Pipeline: generate catalogs → render sources → PSF convolve → add noise.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from skysim.config import SimConfig
from skysim.coordinates import TileInfo
from skysim.layers.base import Catalog, Layer
from skysim.models.morphology import (
    STAMP_BUCKETS,
    _assign_bucket,
    _stamp_sizes_vectorized,
    add_stamp_to_image,
    add_stamps_scatter,
    is_point_source,
    make_sersic_stamp,
    make_sersic_stamps_batch,
    stamp_size_for_galaxy,
)
from skysim.models.psf import get_psf_kernel
from skysim.seed import layer_key, tile_key, master_key
from skysim.telescope.noise import add_noise
from skysim.utils.cosmology import luminosity_distances, luminosity_distances_jax
from skysim.utils.image import add_point_source, add_point_sources_batch, fft_convolve2d


def _re_kpc_to_pix(
    log_re_kpc: jnp.ndarray,
    z: jnp.ndarray,
    pixel_scale_arcsec: float,
) -> jnp.ndarray:
    """Convert physical half-light radius to pixels."""
    from skysim.utils.cosmology import angular_diameter_distances_jax
    d_A_kpc = angular_diameter_distances_jax(z) * 1e3
    re_kpc = 10.0 ** log_re_kpc
    re_arcsec = (re_kpc / d_A_kpc) * 206265.0
    re_pix = re_arcsec / pixel_scale_arcsec
    return re_pix


def _flux_to_electrons(
    log_lnu: jnp.ndarray,
    z: jnp.ndarray,
    dl_mpc: jnp.ndarray,
    telescope_area_m2: float,
    exposure_time_s: float,
    filter_bandwidth_hz: float = 1e13,
) -> jnp.ndarray:
    """Convert L_nu to detected electrons (approximate)."""
    dl_cm = dl_mpc * 3.0856776e24
    log_fnu = log_lnu + jnp.log10(1.0 + z) - jnp.log10(4.0 * jnp.pi) - 2.0 * jnp.log10(dl_cm)
    fnu = 10.0 ** log_fnu
    flux = fnu * filter_bandwidth_hz
    photon_energy = 2e-12  # erg (approximate for NIR)
    collecting_area_cm2 = telescope_area_m2 * 1e4
    photon_rate = flux * collecting_area_cm2 / photon_energy
    electrons = photon_rate * exposure_time_s * 0.8
    return electrons


def _log_fnu_to_electrons(
    log_fnu: jnp.ndarray,
    telescope_area_m2: float,
    exposure_time_s: float,
    filter_bandwidth_hz: float = 1e13,
) -> jnp.ndarray:
    """Convert log10(f_nu) [erg/s/cm^2/Hz] to detected electrons."""
    fnu = 10.0 ** log_fnu
    flux = fnu * filter_bandwidth_hz
    photon_energy = 2e-12
    collecting_area_cm2 = telescope_area_m2 * 1e4
    photon_rate = flux * collecting_area_cm2 / photon_energy
    electrons = photon_rate * exposure_time_s * 0.8
    return electrons


def _catalog_to_pixel_coords(
    catalog: Catalog,
    npix: int,
    pixel_scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert RA/Dec to pixel coordinates centered on the image."""
    ra_center = jnp.mean(catalog["ra"])
    dec_center = jnp.mean(catalog["dec"])
    dx_arcsec = (catalog["ra"] - ra_center) * 3600.0 * jnp.cos(jnp.deg2rad(dec_center))
    dy_arcsec = (catalog["dec"] - dec_center) * 3600.0
    px = dx_arcsec / pixel_scale + npix / 2.0
    py = dy_arcsec / pixel_scale + npix / 2.0
    return px, py


def render_image(
    layers: List[Layer],
    tile: TileInfo,
    config: SimConfig,
    psf_type: str = "gaussian",
    psf_fwhm_arcsec: float = 0.1,
    mag_limit: float = 30.0,
    psf_dir: Optional[str] = None,
    psf_pixel_scale: Optional[float] = None,
) -> Dict[str, jnp.ndarray]:
    """Render a complete sky image for one tile.

    Parameters
    ----------
    layers : list of Layer
        Sky layers to include (e.g. [GalaxyLayer(), StarLayer()]).
    tile : TileInfo
        Tile to render.
    config : SimConfig
        Simulation config.
    psf_type : str
        ``"gaussian"``, ``"moffat"``, or ``"file"`` (load from FITS).
    psf_fwhm_arcsec : float
        PSF FWHM in arcseconds (ignored when *psf_type* is ``"file"``).
    mag_limit : float
        Faintest magnitude to render (skip fainter objects).
    psf_dir : str, optional
        Directory containing PSF FITS files (for ``psf_type="file"``).
    psf_pixel_scale : float, optional
        Pixel scale of the FITS PSF in arcsec/pixel (read from header if None).

    Returns
    -------
    dict with keys:
        "noiseless": image before noise
        "image": final noisy image
        "catalogs": dict of layer_name → Catalog
    """
    tel = config.telescope
    npix = tel.npix
    pixel_scale = tel.pixel_scale

    # PSF kernel
    psf_fwhm_pix = psf_fwhm_arcsec / pixel_scale
    psf_kernel = get_psf_kernel(
        psf_type, config.active_filter, pixel_scale, psf_fwhm_pix,
        psf_dir=psf_dir, psf_pixel_scale=psf_pixel_scale,
    )

    # Generate catalogs
    master = master_key(config.seed)
    tk = tile_key(master, tile.tile_index)
    catalogs = {}
    noiseless = jnp.zeros((npix, npix), dtype=jnp.float32)

    # --- Generate LSS density field if requested ---
    density_field = None
    lss_box_mpc = 200.0
    if "lss" in config.layers:
        from skysim.layers.lss import growth_factor_approx, zeldovich_displacement
        lss_key = layer_key(tk, "lss")
        # Use growth factor at median redshift
        z_med = float(0.5 * (config.z_min + config.z_max))
        gf = growth_factor_approx(z_med)
        density_field = zeldovich_displacement(
            lss_key, ngrid=64, box_size_mpc=lss_box_mpc, growth_factor=gf,
        )

    for layer in layers:
        lk = layer_key(tk, layer.name)

        # Pass density field to galaxy layer
        if hasattr(layer, 'name') and layer.name == "galaxies" and density_field is not None:
            catalog = layer.generate_catalog(
                lk, tile, config,
                density_field=density_field,
                density_box_mpc=lss_box_mpc,
            )
        else:
            catalog = layer.generate_catalog(lk, tile, config)

        catalogs[layer.name] = catalog

        if "mag" not in catalog or len(catalog["mag"]) == 0:
            continue

        # Dispatch to the right renderer based on catalog contents
        if "log_lnu" in catalog:
            noiseless = _render_galaxy_catalog(
                noiseless, catalog, config, psf_fwhm_pix, mag_limit,
            )
        elif "log_fnu" in catalog:
            noiseless = _render_star_catalog(
                noiseless, catalog, config, mag_limit,
            )

    # PSF convolution (single pass over the full image)
    convolved = fft_convolve2d(noiseless, psf_kernel)

    # Add noise
    noise_key = jax.random.fold_in(tk, 999)
    noisy = add_noise(noise_key, convolved, tel, config.active_filter)

    return {
        "noiseless": convolved,
        "image": noisy,
        "catalogs": catalogs,
    }


def _render_galaxy_catalog(
    image: jnp.ndarray,
    catalog: Catalog,
    config: SimConfig,
    psf_fwhm_pix: float,
    mag_limit: float,
) -> jnp.ndarray:
    """Render galaxies from a catalog onto the image.

    For each resolved galaxy, renders separate bulge and disc Sersic
    components when bulge_to_total data is available. Point-source
    galaxies (~90%) are batched into a single scatter-add.
    """
    tel = config.telescope
    npix = tel.npix
    pixel_scale = tel.pixel_scale

    mag = catalog["mag"]
    log_lnu = catalog["log_lnu"]
    z = catalog["z"]

    dl_mpc = luminosity_distances_jax(z)
    tel_area = jnp.pi * (tel.aperture_m / 2.0) ** 2
    electrons = _flux_to_electrons(
        log_lnu, z, dl_mpc, float(tel_area), tel.exposure_time_s,
    )

    # Use the total (composite) R_e for point-source classification
    re_pix = _re_kpc_to_pix(catalog["log_re_kpc"], z, pixel_scale)
    px, py = _catalog_to_pixel_coords(catalog, npix, pixel_scale)
    is_point = is_point_source(re_pix, psf_fwhm_pix) | (re_pix < 1.0)
    bright_enough = mag < mag_limit

    # --- Batch render all point sources at once ---
    point_mask = is_point & bright_enough
    image = add_point_sources_batch(
        image,
        jnp.where(point_mask, px, -1.0),
        jnp.where(point_mask, py, -1.0),
        jnp.where(point_mask, electrons, 0.0),
    )

    # --- Render resolved galaxies (bulge + disc components) ---
    resolved_mask = (~is_point) & bright_enough
    on_image = (px > -50) & (px < npix + 50) & (py > -50) & (py < npix + 50)
    resolved_mask = resolved_mask & on_image & (electrons > 0)

    n_resolved = int(jnp.sum(resolved_mask))
    if n_resolved == 0:
        return image

    # Cap at 20K resolved galaxies (brightest first)
    if n_resolved > 20000:
        resolved_mags = jnp.where(resolved_mask, mag, 99.0)
        order = jnp.argsort(resolved_mags)
        cutoff_mag = float(mag[order[19999]])
        resolved_mask = resolved_mask & (mag <= cutoff_mag)
        n_resolved = min(n_resolved, 20000)

    ridx = jnp.where(resolved_mask, size=n_resolved, fill_value=0)[0]

    # Check if bulge+disc data is available
    has_bd = "bulge_to_total" in catalog and "log_re_bulge_kpc" in catalog
    if has_bd:
        image = _render_bulge_disc(
            image, catalog, ridx, z, electrons, px, py,
            pixel_scale, npix,
        )
    else:
        # Fallback: single-component rendering
        image = _render_single_component(
            image, catalog, ridx, re_pix, electrons, px, py,
        )

    return image


def _render_bulge_disc(
    image: jnp.ndarray,
    catalog: Catalog,
    ridx: jnp.ndarray,
    z: jnp.ndarray,
    electrons: jnp.ndarray,
    px: jnp.ndarray,
    py: jnp.ndarray,
    pixel_scale: float,
    npix: int,
) -> jnp.ndarray:
    """Render resolved galaxies as bulge + disc components.

    Creates two entries per galaxy (bulge and disc) and renders them
    together through the batched stamp pipeline.
    """
    bt = catalog["bulge_to_total"][ridx]
    pa = catalog["pa"][ridx]
    flux = electrons[ridx]
    x = px[ridx].astype(jnp.int32)
    y = py[ridx].astype(jnp.int32)

    # --- Bulge parameters ---
    re_bulge_pix = _re_kpc_to_pix(catalog["log_re_bulge_kpc"][ridx], z[ridx], pixel_scale)
    n_bulge = jnp.full(len(ridx), 4.0)  # de Vaucouleurs
    q_bulge = catalog["axis_ratio_bulge"][ridx]
    flux_bulge = flux * bt

    # --- Disc parameters ---
    re_disc_pix = _re_kpc_to_pix(catalog["log_re_disc_kpc"][ridx], z[ridx], pixel_scale)
    n_disc = jnp.full(len(ridx), 1.0)  # exponential
    q_disc = catalog["axis_ratio"][ridx]
    flux_disc = flux * (1.0 - bt)

    # Concatenate bulge and disc into a single batch
    all_n = jnp.concatenate([n_bulge, n_disc])
    all_re = jnp.concatenate([re_bulge_pix, re_disc_pix])
    all_q = jnp.concatenate([q_bulge, q_disc])
    all_pa = jnp.concatenate([pa, pa])
    all_flux = jnp.concatenate([flux_bulge, flux_disc])
    all_x = jnp.concatenate([x, x])
    all_y = jnp.concatenate([y, y])

    # Filter out zero-flux components (e.g. B/T=0 means no bulge)
    valid = all_flux > 0
    valid_idx = jnp.where(valid, size=int(jnp.sum(valid)), fill_value=0)[0]

    r_n = all_n[valid_idx]
    r_re = all_re[valid_idx]
    r_q = all_q[valid_idx]
    r_pa = all_pa[valid_idx]
    r_flux = all_flux[valid_idx]
    r_px = all_x[valid_idx]
    r_py = all_y[valid_idx]

    # Compute stamp sizes and render by bucket
    stamp_sizes = _stamp_sizes_vectorized(r_re, r_n)
    buckets = _assign_bucket(stamp_sizes)

    for bsize in STAMP_BUCKETS:
        bmask = buckets == bsize
        n_in_bucket = int(jnp.sum(bmask))
        if n_in_bucket == 0:
            continue

        bidx = jnp.where(bmask, size=n_in_bucket, fill_value=0)[0]
        stamps = make_sersic_stamps_batch(
            r_n[bidx], r_re[bidx], r_q[bidx], r_pa[bidx],
            r_flux[bidx], bsize,
        )
        image = add_stamps_scatter(
            image, stamps, bsize, r_px[bidx], r_py[bidx],
        )

    return image


def _render_single_component(
    image: jnp.ndarray,
    catalog: Catalog,
    ridx: jnp.ndarray,
    re_pix: jnp.ndarray,
    electrons: jnp.ndarray,
    px: jnp.ndarray,
    py: jnp.ndarray,
) -> jnp.ndarray:
    """Fallback single-Sersic rendering (no bulge+disc data)."""
    r_n = catalog["sersic_n"][ridx]
    r_q = catalog["axis_ratio"][ridx]
    r_pa = catalog["pa"][ridx]
    r_re = re_pix[ridx]
    r_flux = electrons[ridx]
    r_px = px[ridx].astype(jnp.int32)
    r_py = py[ridx].astype(jnp.int32)

    stamp_sizes = _stamp_sizes_vectorized(r_re, r_n)
    buckets = _assign_bucket(stamp_sizes)

    for bsize in STAMP_BUCKETS:
        bmask = buckets == bsize
        n_in_bucket = int(jnp.sum(bmask))
        if n_in_bucket == 0:
            continue

        bidx = jnp.where(bmask, size=n_in_bucket, fill_value=0)[0]
        stamps = make_sersic_stamps_batch(
            r_n[bidx], r_re[bidx], r_q[bidx], r_pa[bidx],
            r_flux[bidx], bsize,
        )
        image = add_stamps_scatter(
            image, stamps, bsize, r_px[bidx], r_py[bidx],
        )

    return image


def _render_star_catalog(
    image: jnp.ndarray,
    catalog: Catalog,
    config: SimConfig,
    mag_limit: float,
) -> jnp.ndarray:
    """Render stars as point sources onto the image (vectorized)."""
    tel = config.telescope
    log_fnu = catalog["log_fnu"]
    mag = catalog["mag"]

    tel_area = jnp.pi * (tel.aperture_m / 2.0) ** 2
    electrons = _log_fnu_to_electrons(
        log_fnu, float(tel_area), tel.exposure_time_s,
    )

    npix = tel.npix
    px, py = _catalog_to_pixel_coords(catalog, npix, tel.pixel_scale)
    bright_enough = mag < mag_limit

    image = add_point_sources_batch(
        image,
        jnp.where(bright_enough, px, -1.0),
        jnp.where(bright_enough, py, -1.0),
        jnp.where(bright_enough, electrons, 0.0),
    )
    return image
