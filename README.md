# SkySim

Fast, portable sky image simulator for generating realistic broadband optical/IR mock observations. Built on JAX for deterministic, GPU-acceleratable rendering.

SkySim generates scientifically grounded mock sky images with galaxies, stars, and telescope effects — from catalog generation through to noisy detector images — in seconds on a single CPU.

## Example of Web UI

Mock JWST RGB star field near the galactic center generated using SkySim. Try it here: https://skysim.tharvey.space

<img width="1440" height="723" alt="image" src="https://github.com/user-attachments/assets/a2449176-2fab-43c5-a107-a300fab609e2" />


## Features

- **Galaxy populations** from a double-Schechter GSMF (Weaver+23) with redshift evolution, mass-size (van der Wel+14), mass-metallicity (Zahid+14), and parametric SFH models
- **SPS Modelling with Synthesizer** Configurable SPS modelling using Synthesizer with realistic SFHs, dust reprocessing, nebular emission, all via [Synthesizer](https://synthesizer-project.github.io/synthesizer/)
- **Two-component Sersic morphologies** rendered as 2-D stamps with ellipticity and position angle, with varying bulge-to-disk ratios, batched with `jax.vmap`
- **Stellar foreground** from a 4-component Milky Way model (thin/thick disc, halo, bulge)
- **Large-scale structure** via Zel'dovich approximation density field modulation
- **PSF convolution** — Gaussian, Moffat, or file-based (e.g. WebbPSF FITS files)
- **Realistic noise** — Poisson photon noise, read noise, dark current, sky background
- **23 broadband filters** — JWST NIRCam, HST ACS/WFC3, Rubin/LSST ugrizy, Euclid VIS/NISP, and more (any SVO-compatible filter can be added)
- **4 telescope presets** — JWST NIRCam, HST ACS, Rubin LSST, Euclid VIS (custom configs supported)
- **Deterministic** — same seed + sky position = bit-identical output via hierarchical JAX PRNG
- **Web UI** — React frontend with interactive pan/zoom, stretch controls, RGB compositing, and filter selection
- **Docker** support for containerised deployment

## Quick Start

```bash
# Install
pip install -e ".[api,dev]"

# Generate photometry lookup table (one-time, requires Synthesizer)
python -m skysim.scripts.generate_tables

# Run demo — renders a JWST NIRCam tile and saves a FITS file
python -m skysim.scripts.demo
```

## Web UI

```bash
# Build the frontend
cd web && npm install && npm run build && cd ..

# Start the server (serves API + frontend)
uvicorn skysim.api.server:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

For development with hot-reload:

```bash
# Terminal 1: API server
uvicorn skysim.api.server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Vite dev server (proxies /api to :8000)
cd web && npm run dev
```

## Docker

```bash
docker compose up --build
# Open http://localhost:8000
```

## File-Based PSFs

Drop PSF FITS files into `skysim/data/psfs/` with the naming convention:

```
JWST_NIRCam.F444W.fits    # for filter JWST/NIRCam.F444W
HST_ACS_WFC.F814W.fits    # for filter HST/ACS_WFC.F814W
```

The pixel scale is read from the FITS header (`PIXELSCL` or `CDELT2`) and the kernel is resampled to match the image pixel scale. Select `psf_type="file"` in the API or "File (FITS)" in the web UI.

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/render/raw` | Render image, return raw float32 binary |
| `GET /api/render/png` | Render image, return stretched PNG |
| `GET /api/catalog` | Generate galaxy catalog as JSON |
| `GET /api/filters` | List available filters |
| `GET /api/telescopes` | List telescope presets |
| `GET /api/psfs` | List available file-based PSFs |
| `GET /api/health` | Health check |

## Project Structure

```
skysim/
  config.py              # Dataclass configuration, telescope presets
  seed.py                # Hierarchical deterministic seeding (JAX PRNG)
  coordinates.py         # HEALPix tiling, coordinate utilities
  layers/
    galaxies.py          # Galaxy population synthesis pipeline
    stars.py             # MW stellar foreground
    lss.py               # Zel'dovich density field
  models/
    schechter.py         # Double-Schechter GSMF (Weaver+23)
    mass_size.py         # van der Wel+14 mass-size relation
    mass_metallicity.py  # Zahid+14 MZR
    sfh.py               # Parametric SFH models
    morphology.py        # Sersic profile rendering (batched)
    photometry.py        # SED lookup table with dust axis
    stellar_model.py     # MW stellar density model
    psf.py               # Gaussian, Moffat, and FITS PSF kernels
  telescope/
    renderer.py          # Image assembly pipeline
    noise.py             # Detector noise model
  data/
    psfs/                # Drop PSF FITS files here
    seds/                # Precomputed photometry table
  api/
    server.py            # FastAPI server
  scripts/
    generate_tables.py   # Build photometry lookup from Synthesizer
    demo.py              # CLI demo
web/                     # React frontend
```

## Performance

Benchmarked on a single CPU (Apple M-series), rendering a 4258x4258 JWST NIRCam image with ~37,000 galaxies:

| Optimisation | Time |
|---|---|
| Baseline | 112 s |
| + Precomputed cosmology lookups | 71 s |
| + Vectorised point-source rendering | 59 s |
| + Batched Sersic stamps (vmap + scatter) | 14 s |

## Dependencies

- **Runtime**: `jax`, `jaxlib`, `numpy`, `astropy`, `healpy`
- **API**: `fastapi`, `uvicorn`, `pillow`
- **Table generation**: `scipy`, `synthesizer`
- **Frontend**: Node.js 18+ (build only)
