"""Quick demo: render a sky image and save to disk.

Usage:
    python -m skysim.scripts.demo [--ra 180] [--dec 0] [--seed 42]
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="SkySim demo renderer")
    parser.add_argument("--ra", type=float, default=180.0, help="RA (deg)")
    parser.add_argument("--dec", type=float, default=0.0, help="Dec (deg)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--telescope", type=str, default="jwst_nircam",
                        choices=["jwst_nircam", "hst_acs", "rubin_lsst", "euclid_vis"])
    parser.add_argument("--filter", type=str, default="JWST/NIRCam.F200W")
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--mag-limit", type=float, default=28.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.1, help="PSF FWHM (arcsec)")
    parser.add_argument("--output", type=str, default="skysim_demo.fits")
    parser.add_argument("--no-stars", action="store_true")
    args = parser.parse_args()

    from skysim.config import (
        EUCLID_VIS,
        HST_ACS,
        JWST_NIRCAM,
        RUBIN_LSST,
        SimConfig,
    )
    from skysim.coordinates import TileInfo, radec_to_tile
    from skysim.layers.galaxies import GalaxyLayer
    from skysim.layers.stars import StarLayer
    from skysim.telescope.renderer import render_image

    presets = {
        "jwst_nircam": JWST_NIRCAM,
        "hst_acs": HST_ACS,
        "rubin_lsst": RUBIN_LSST,
        "euclid_vis": EUCLID_VIS,
    }
    telescope = presets[args.telescope]

    config = SimConfig(
        seed=args.seed,
        nside=args.nside,
        telescope=telescope,
        active_filter=args.filter,
    )

    tile_idx = radec_to_tile(config.nside, args.ra, args.dec)
    tile = TileInfo.from_index(config.nside, tile_idx)

    print(f"SkySim Demo")
    print(f"  RA, Dec:    {args.ra:.1f}, {args.dec:.1f}")
    print(f"  Tile:       {tile_idx} (area={tile.area_arcmin2:.1f} arcmin²)")
    print(f"  Telescope:  {telescope.name}")
    print(f"  Filter:     {args.filter}")
    print(f"  Image size: {telescope.npix}×{telescope.npix}")
    print(f"  Mag limit:  {args.mag_limit}")

    layers = [GalaxyLayer()]
    if not args.no_stars:
        layers.append(StarLayer())

    print("Rendering...")
    t0 = time.time()
    result = render_image(
        layers=layers,
        tile=tile,
        config=config,
        psf_fwhm_arcsec=args.psf_fwhm,
        mag_limit=args.mag_limit,
    )
    dt = time.time() - t0

    img = result["image"]
    n_gals = len(result["catalogs"].get("galaxies", {}).get("mag", []))
    n_stars = len(result["catalogs"].get("stars", {}).get("mag", []))

    print(f"  Galaxies:   {n_gals}")
    print(f"  Stars:      {n_stars}")
    print(f"  Render time: {dt:.1f}s")

    # Save output
    img_np = np.array(img)
    if args.output.endswith(".fits"):
        from astropy.io import fits
        hdu = fits.PrimaryHDU(img_np)
        hdu.header["RA"] = args.ra
        hdu.header["DEC"] = args.dec
        hdu.header["SEED"] = args.seed
        hdu.header["FILTER"] = args.filter
        hdu.header["TELESCOP"] = telescope.name
        hdu.header["PIXSCALE"] = telescope.pixel_scale
        hdu.header["EXPTIME"] = telescope.exposure_time_s
        hdu.header["NGAL"] = n_gals
        hdu.header["NSTAR"] = n_stars
        hdu.writeto(args.output, overwrite=True)
    else:
        np.save(args.output, img_np)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
