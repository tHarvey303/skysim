"""FastAPI server for SkySim.

Endpoints under /api:
    GET  /api/health         — Health check
    GET  /api/filters        — List available filters
    GET  /api/telescopes     — List available telescope presets
    GET  /api/render/png     — Render single-band image as PNG
    GET  /api/render/raw     — Render single-band, return float32 binary
    GET  /api/catalog        — Generate and return a galaxy catalog as JSON

In production the built React frontend is served from web/dist/ at /.

Usage:
    uvicorn skysim.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
from fastapi import APIRouter, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from skysim.config import (
    EUCLID_VIS,
    HST_ACS,
    JWST_NIRCAM,
    RUBIN_LSST,
    SimConfig,
    TelescopeConfig,
)
from skysim.coordinates import TileInfo
from skysim.layers.galaxies import GalaxyLayer
from skysim.layers.stars import StarLayer
from skysim.telescope.renderer import render_image

app = FastAPI(
    title="SkySim",
    description="Fast sky image simulator",
    version="0.1.0",
)

# CORS for development (Vite dev server on :5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Image-Width", "X-Image-Height", "X-Render-Time",
                     "X-Galaxy-Count", "X-Star-Count",
                     "X-RA-Center", "X-Dec-Center", "X-Pixel-Scale"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

TELESCOPE_PRESETS = {
    "jwst_nircam": JWST_NIRCAM,
    "hst_acs": HST_ACS,
    "rubin_lsst": RUBIN_LSST,
    "euclid_vis": EUCLID_VIS,
}

AVAILABLE_FILTERS = [
    "JWST/NIRCam.F090W", "JWST/NIRCam.F115W", "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W", "JWST/NIRCam.F277W", "JWST/NIRCam.F356W",
    "JWST/NIRCam.F444W",
    "HST/ACS_WFC.F435W", "HST/ACS_WFC.F606W", "HST/ACS_WFC.F814W",
    "HST/WFC3_IR.F105W", "HST/WFC3_IR.F125W", "HST/WFC3_IR.F160W",
    "Euclid/VIS.vis", "Euclid/NISP.H", "Euclid/NISP.J", "Euclid/NISP.Y",
    "LSST/LSST.u", "LSST/LSST.g", "LSST/LSST.r",
    "LSST/LSST.i", "LSST/LSST.z", "LSST/LSST.y",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(
    seed: int,
    telescope: str,
    filter_code: str,
    nside: int,
    fov_arcmin: Optional[float],
    exposure_time_s: Optional[float],
) -> SimConfig:
    tel = TELESCOPE_PRESETS.get(telescope, JWST_NIRCAM)
    overrides = {}
    if fov_arcmin is not None:
        overrides["fov_arcmin"] = fov_arcmin
    if exposure_time_s is not None:
        overrides["exposure_time_s"] = exposure_time_s
    if overrides:
        tel = TelescopeConfig(
            name=tel.name,
            pixel_scale=tel.pixel_scale,
            fov_arcmin=overrides.get("fov_arcmin", tel.fov_arcmin),
            aperture_m=tel.aperture_m,
            read_noise_e=tel.read_noise_e,
            dark_current_e_s=tel.dark_current_e_s,
            exposure_time_s=overrides.get("exposure_time_s", tel.exposure_time_s),
        )
    return SimConfig(seed=seed, nside=nside, telescope=tel, active_filter=filter_code)


def _do_render(
    ra: float,
    dec: float,
    seed: int,
    telescope: str,
    filter_code: str,
    nside: int,
    fov_arcmin: Optional[float],
    exposure_time_s: Optional[float],
    mag_limit: float,
    psf_fwhm: float,
    psf_type: str,
    include_stars: bool,
):
    """Run the rendering pipeline and return (result_dict, config, tile)."""
    import time

    from skysim.coordinates import radec_to_tile

    config = _build_config(seed, telescope, filter_code, nside, fov_arcmin, exposure_time_s)
    tile_idx = radec_to_tile(config.nside, ra, dec)
    tile = TileInfo.from_index(config.nside, tile_idx)

    layers = [GalaxyLayer()]
    if include_stars:
        layers.append(StarLayer())

    t0 = time.perf_counter()
    result = render_image(
        layers=layers,
        tile=tile,
        config=config,
        psf_type=psf_type,
        psf_fwhm_arcsec=psf_fwhm,
        mag_limit=mag_limit,
    )
    dt = time.perf_counter() - t0

    n_gal = len(result["catalogs"].get("galaxies", {}).get("mag", []))
    n_star = len(result["catalogs"].get("stars", {}).get("mag", []))

    return result, config, tile, dt, n_gal, n_star

# ---------------------------------------------------------------------------
# API Router
# ---------------------------------------------------------------------------

api = APIRouter(prefix="/api")


@api.get("/psfs")
def list_psfs():
    """List filter codes that have a file-based PSF available."""
    from skysim.models.psf import list_available_psfs
    return {"psfs": list_available_psfs()}




@api.get("/health")
def health():
    return {"status": "ok"}


@api.get("/filters")
def list_filters():
    return {"filters": AVAILABLE_FILTERS}


@api.get("/telescopes")
def list_telescopes():
    return {
        name: {
            "name": t.name,
            "pixel_scale": t.pixel_scale,
            "fov_arcmin": t.fov_arcmin,
            "aperture_m": t.aperture_m,
            "read_noise_e": t.read_noise_e,
            "dark_current_e_s": t.dark_current_e_s,
            "exposure_time_s": t.exposure_time_s,
        }
        for name, t in TELESCOPE_PRESETS.items()
    }


# Common query parameters
_Q_RA = Query(180.0, description="RA in degrees")
_Q_DEC = Query(0.0, description="Dec in degrees")
_Q_SEED = Query(42, description="Random seed")
_Q_TEL = Query("jwst_nircam", description="Telescope preset")
_Q_FILTER = Query("JWST/NIRCam.F200W", description="Filter code")
_Q_NSIDE = Query(256, description="HEALPix NSIDE")
_Q_FOV = Query(None, description="Override FoV (arcmin)")
_Q_EXP = Query(None, description="Override exposure time (s)")
_Q_MAG = Query(28.0, description="Magnitude limit")
_Q_PSF = Query(0.1, description="PSF FWHM (arcsec)")
_Q_PSF_TYPE = Query("gaussian", description="PSF type: gaussian, moffat, or file")
_Q_STARS = Query(True, description="Include stars")


@api.get("/render/raw")
def render_raw(
    ra: float = _Q_RA,
    dec: float = _Q_DEC,
    seed: int = _Q_SEED,
    telescope: str = _Q_TEL,
    filter_code: str = _Q_FILTER,
    nside: int = _Q_NSIDE,
    fov_arcmin: Optional[float] = _Q_FOV,
    exposure_time_s: Optional[float] = _Q_EXP,
    mag_limit: float = _Q_MAG,
    psf_fwhm: float = _Q_PSF,
    psf_type: str = _Q_PSF_TYPE,
    include_stars: bool = _Q_STARS,
):
    """Render and return raw float32 image data as binary.

    Response is a binary blob: the image as row-major float32.
    Image dimensions are in X-Image-Width / X-Image-Height headers.
    """
    result, config, tile, dt, n_gal, n_star = _do_render(
        ra, dec, seed, telescope, filter_code, nside,
        fov_arcmin, exposure_time_s, mag_limit, psf_fwhm, psf_type, include_stars,
    )

    img = np.array(result["image"], dtype=np.float32)
    h, w = img.shape

    return Response(
        content=img.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Image-Width": str(w),
            "X-Image-Height": str(h),
            "X-Render-Time": f"{dt:.2f}",
            "X-Galaxy-Count": str(n_gal),
            "X-Star-Count": str(n_star),
            "X-RA-Center": f"{tile.ra_center:.6f}",
            "X-Dec-Center": f"{tile.dec_center:.6f}",
            "X-Pixel-Scale": f"{config.telescope.pixel_scale:.6f}",
        },
    )


@api.get("/render/png")
def render_png(
    ra: float = _Q_RA,
    dec: float = _Q_DEC,
    seed: int = _Q_SEED,
    telescope: str = _Q_TEL,
    filter_code: str = _Q_FILTER,
    nside: int = _Q_NSIDE,
    fov_arcmin: Optional[float] = _Q_FOV,
    exposure_time_s: Optional[float] = _Q_EXP,
    mag_limit: float = _Q_MAG,
    psf_fwhm: float = _Q_PSF,
    psf_type: str = _Q_PSF_TYPE,
    include_stars: bool = _Q_STARS,
    stretch: str = Query("asinh", description="Stretch: linear, sqrt, log, asinh"),
    pmin: float = Query(1.0, description="Lower percentile clip"),
    pmax: float = Query(99.5, description="Upper percentile clip"),
):
    """Render and return a stretched PNG."""
    result, config, tile, dt, n_gal, n_star = _do_render(
        ra, dec, seed, telescope, filter_code, nside,
        fov_arcmin, exposure_time_s, mag_limit, psf_fwhm, psf_type, include_stars,
    )

    img = np.array(result["image"])
    img = _apply_stretch(img, stretch, pmin, pmax)

    buf = io.BytesIO()
    _save_png(buf, img)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Render-Time": f"{dt:.2f}",
            "X-Galaxy-Count": str(n_gal),
            "X-Star-Count": str(n_star),
        },
    )


@api.get("/render/fits")
def render_fits(
    ra: float = _Q_RA,
    dec: float = _Q_DEC,
    seed: int = _Q_SEED,
    telescope: str = _Q_TEL,
    filter_code: str = _Q_FILTER,
    nside: int = _Q_NSIDE,
    fov_arcmin: Optional[float] = _Q_FOV,
    exposure_time_s: Optional[float] = _Q_EXP,
    mag_limit: float = _Q_MAG,
    psf_fwhm: float = _Q_PSF,
    psf_type: str = _Q_PSF_TYPE,
    include_stars: bool = _Q_STARS,
    noiseless: bool = Query(False, description="Return noiseless image instead"),
):
    """Render and return a FITS file with WCS headers."""
    from astropy.io import fits
    from astropy.wcs import WCS

    result, config, tile, dt, n_gal, n_star = _do_render(
        ra, dec, seed, telescope, filter_code, nside,
        fov_arcmin, exposure_time_s, mag_limit, psf_fwhm, psf_type, include_stars,
    )

    img_key = "noiseless" if noiseless else "image"
    img = np.array(result[img_key], dtype=np.float32)
    h, w = img.shape

    # Build WCS
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = [tile.ra_center, tile.dec_center]
    pixel_scale_deg = config.telescope.pixel_scale / 3600.0
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]

    header = wcs.to_header()
    header["TELESCOP"] = config.telescope.name
    header["FILTER"] = filter_code
    header["EXPTIME"] = config.telescope.exposure_time_s
    header["PIXSCALE"] = (config.telescope.pixel_scale, "arcsec/pixel")
    header["SEED"] = config.seed
    header["NGAL"] = n_gal
    header["NSTAR"] = n_star
    header["RENDTIME"] = (round(dt, 2), "Render time (s)")

    hdu = fits.PrimaryHDU(data=img, header=header)
    buf = io.BytesIO()
    hdu.writeto(buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/fits",
        headers={
            "Content-Disposition": f"attachment; filename=skysim_{filter_code.replace('/', '_')}.fits",
            "X-Render-Time": f"{dt:.2f}",
            "X-Galaxy-Count": str(n_gal),
            "X-Star-Count": str(n_star),
        },
    )


@api.get("/catalog")
def catalog(
    ra: float = Query(180.0),
    dec: float = Query(0.0),
    seed: int = Query(42),
    nside: int = Query(256),
    filter_code: str = Query("JWST/NIRCam.F200W"),
    mag_limit: float = Query(28.0),
    max_objects: int = Query(10000),
):
    """Generate and return a galaxy catalog as JSON."""
    import jax

    from skysim.coordinates import radec_to_tile
    from skysim.seed import layer_key, master_key, tile_key

    config = SimConfig(seed=seed, nside=nside, active_filter=filter_code)
    tile_idx = radec_to_tile(nside, ra, dec)
    tile = TileInfo.from_index(nside, tile_idx)

    key = layer_key(tile_key(master_key(seed), tile_idx), "galaxies")
    layer = GalaxyLayer()
    cat = layer.generate_catalog(key, tile, config)

    mag = np.array(cat["mag"])
    mask = mag < mag_limit
    indices = np.where(mask)[0][:max_objects]

    result = {}
    for col, arr in cat.items():
        vals = np.array(arr)
        if vals.ndim == 1 and len(vals) > 0:
            result[col] = vals[indices].tolist()

    return {
        "tile_index": tile_idx,
        "ra_center": tile.ra_center,
        "dec_center": tile.dec_center,
        "n_total": int(mask.sum()),
        "n_returned": len(indices),
        "catalog": result,
    }


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------


def _apply_stretch(img: np.ndarray, stretch: str, pmin: float, pmax: float) -> np.ndarray:
    """Apply stretch and normalize to uint8."""
    positive = img[img > 0]
    if len(positive) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    if stretch == "asinh":
        scale = np.percentile(positive, 50)
        img = np.arcsinh(img / max(scale, 1e-30) * 10.0)
    elif stretch == "sqrt":
        img = np.sqrt(np.maximum(img, 0.0))
    elif stretch == "log":
        img = np.log10(np.maximum(img, 1e-30))

    vmin = np.percentile(img, pmin)
    vmax = np.percentile(img, pmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return (img * 255).astype(np.uint8)


def _save_png(buf: io.BytesIO, img: np.ndarray):
    """Save a 2-D uint8 array as PNG."""
    try:
        from PIL import Image
        Image.fromarray(img, mode="L").save(buf, format="PNG")
    except ImportError:
        buf.write(img.tobytes())


# ---------------------------------------------------------------------------
# Mount routes and static files
# ---------------------------------------------------------------------------

app.include_router(api)

# Serve built frontend in production
_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True))


# Keep legacy endpoints for backward compatibility
@app.get("/health")
def health_legacy():
    return health()


@app.get("/filters")
def filters_legacy():
    return list_filters()


@app.get("/telescopes")
def telescopes_legacy():
    return list_telescopes()


@app.get("/render")
def render_legacy(
    ra: float = Query(180.0),
    dec: float = Query(0.0),
    seed: int = Query(42),
    telescope: str = Query("jwst_nircam"),
    filter_code: str = Query("JWST/NIRCam.F200W"),
    nside: int = Query(256),
    fov_arcmin: Optional[float] = Query(None),
    mag_limit: float = Query(28.0),
    psf_fwhm: float = Query(0.1),
    include_stars: bool = Query(True),
):
    return render_png(
        ra=ra, dec=dec, seed=seed, telescope=telescope,
        filter_code=filter_code, nside=nside, fov_arcmin=fov_arcmin,
        mag_limit=mag_limit, psf_fwhm=psf_fwhm, include_stars=include_stars,
    )
