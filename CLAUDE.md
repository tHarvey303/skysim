# SkySim Implementation Plan

## Context
Build a fast, portable, modular sky simulator for generating realistic broadband optical/IR images. This is a greenfield PhD project — no existing code. The tool must produce deterministic, scientifically grounded mock sky images with galaxies, stars, and telescope effects. Near-term web deployment via a FastAPI backend serving rendered images.

## Tech Stack
- **Python 3.10+** with **JAX** as the numerical backend
  - JAX PRNG key-splitting for hierarchical deterministic seeding
  - JIT compilation, `vmap` vectorization, GPU support via XLA
  - `jax.scipy.special.gammainc` for Sersic profiles
- **Supporting**: `numpy`, `scipy` (table precomputation only), `astropy` (coordinates, FITS I/O at boundary)
- **Data**: bundled `.npz` lookup tables (~50 MB total), no runtime network calls
- **API**: FastAPI + uvicorn for serving rendered images
- **Filters**: broad set — JWST NIRCam/MIRI, HST ACS/WFC3, Rubin ugrizy, Euclid VIS/NISP, SDSS, 2MASS, Johnson-Cousins. Can all be obtained using the Synthesizer project (https://synthesizer-project.github.io/synthesizer/) which has Instrument/Filter machinery which can be used to generate the photometry we can precompute on a grid and interpolate at runtime.
- **LSS**: Zel'dovich approximation for cosmic web density field

- Synthesizer guide

- SPS grid is in '/Users/user/Documents/PhD/synthesizer/grids/test_grid.hdf5' and can be loaded following:
https://synthesizer-project.github.io/synthesizer/emission_grids/grids_example.html

Emission models are here: https://synthesizer-project.github.io/synthesizer/emission_models/premade_models/premade_models.html

Star models are here: https://synthesizer-project.github.io/synthesizer/galaxy_components/stars.html

SFH models are here: https://synthesizer-project.github.io/synthesizer/galaxy_components/sfh.html

Filters are here: https://synthesizer-project.github.io/synthesizer/observatories/filters.html

Generating photometry for a single galaxy: https://synthesizer-project.github.io/synthesizer/observables/photometry/galaxy_phot.html

Pipeline is here: https://synthesizer-project.github.io/synthesizer/pipeline/pipeline_example.html


## Package Structure
```
skysim/
  __init__.py
  config.py                # Dataclass-based configuration, filter definitions
  seed.py                  # Hierarchical deterministic seeding (JAX PRNG)
  coordinates.py           # Sky tiling (HEALPix-like), coordinate utils

  layers/
    base.py                # Layer protocol: generate_catalog() + render()
    lss.py                 # Zel'dovich-approximation density field
    galaxies.py            # Galaxy population synthesis pipeline
    stars.py               # MW stellar foreground (disc/halo/bulge)

  models/
    schechter.py           # Double-Schechter GSMF, inverse-CDF sampling
    mass_size.py           # van der Wel+14 mass-size relation
    mass_metallicity.py    # Zahid+14 / Maiolino+08 MZR
    sfh.py                 # Parametric SFH models (tau, delayed-tau, constant)
    morphology.py          # Sersic profile rendering, stamp library, bulge+disc
    photometry.py          # SED lookup table / emulator
    stellar_model.py       # MW star density (Besancon-like simplified)
    psf.py                 # Gaussian, Moffat, Airy PSF models

  telescope/
    instrument.py          # Telescope/instrument config (pixel scale, FoV, filter)
    renderer.py            # Image assembly: layers → PSF convolve → noise
    noise.py               # Poisson + Gaussian + background noise

  data/
    filters/               # Filter transmission curves (.npz)
    seds/                  # SED template grid (.npz)
    isochrones/            # Stellar LF lookup (.npz)

  api/
    server.py              # FastAPI endpoints for rendering images

  utils/
    interpolation.py       # JAX-compatible fast interpolation
    spatial.py             # Spatial indexing on sky
    image.py               # FFT convolution, subpixel rendering

  scripts/
    generate_tables.py     # One-time: build lookup tables from Synthesizer (https://synthesizer-project.github.io/synthesizer/) with e.g. BPASS SPS models. Synthesizer can generate SEDs and photometry for parametric SFH, dust attenuation, dust emission, metallicity, etc., which we can precompute on a grid and interpolate at runtime. There is a Pipeline functionality which allow batching over many galaxy simulations when generating the tables. 
    validate.py            # Compare outputs to observed distributions
    demo.py                # Quick demo

  tests/
    test_seed.py
    test_schechter.py
    test_morphology.py
    test_renderer.py

  pyproject.toml
```

## Implementation Phases

### Phase 1: Skeleton & Seeding
- `pyproject.toml`, package structure, test infrastructure
- `seed.py`: hierarchical PRNG (master → tile → layer → object via `jax.random.fold_in`)
- `coordinates.py`: HEALPix-based tiling (minimal pure implementation)
- `config.py`: dataclass configuration
- `layers/base.py`: Layer protocol
- Tests proving determinism

### Phase 2: Galaxy Population
- `models/schechter.py`: double-Schechter GSMF with redshift evolution (Weaver+23)
- `models/mass_size.py`: van der Wel+14 with 0.15 dex scatter
- `models/mass_metallicity.py`: Zahid+14 MZR
- `models/sfh.py`: parametric SFH models
- `models/photometry.py`: precomputed (mass, z, Z, SFH) → magnitude grid
- `layers/galaxies.py`: full catalog generation pipeline
- `scripts/generate_tables.py`: build SED lookup using Synthesizer (https://synthesizer-project.github.io/) with e.g. BPASS SPS models.
- Validate against observed GSMF, number counts, color distributions

### Phase 3: Morphology & Rendering
- `models/morphology.py`: Sersic profiles (stamp library + analytic fallback)
- `models/psf.py`: Gaussian, Moffat models
- `utils/image.py`: FFT convolution, subpixel shifts
- `telescope/renderer.py`: full image pipeline (compose layers → PSF → noise)
- `telescope/noise.py`: Poisson + read noise + sky background
- Benchmark: target <5s per 4k×4k image on CPU

### Phase 4: Stellar Foreground
- `models/stellar_model.py`: thin disc + thick disc + halo + bulge density
- `layers/stars.py`: star catalog generation, point-source rendering
- Brown dwarf component for IR bands
- Stellar isochrone/LF lookup table

### Phase 5: Large-Scale Structure
- `layers/lss.py`: Zel'dovich approximation density field
- Modulate galaxy number density by LSS field
- Optional HOD-like halo assignment for clustering

### Phase 6: API & Polish
- `api/server.py`: FastAPI endpoints (render image, query catalog, list filters)
- Performance optimization (profiling, JIT coverage, stamp library tuning)
- LOD system: point-source approximation for sub-seeing galaxies
- Documentation, demo notebooks

## Key Architecture Decisions

1. **Struct-of-arrays catalogs**: `dict[str, jnp.ndarray]` not lists of objects — enables JAX vectorization
2. **Single PSF convolution**: convolve summed image once, not per-object
3. **Stamp library for Sersic**: precomputed at discrete (n, R_e, q) values; nearest-neighbor + scale + rotate at render time
4. **Photometry emulator**: precomputed N-D grid, runtime interpolation only — no SPS at runtime
5. **Tile-based generation**: any sky tile reproducible independently without generating neighbors
6. **Avoid hard-coding details**: config-driven (e.g. filter choice, noise levels) for flexibility
  and for e.g. changing the underliyng mass function or size relation without changing code, just regenerating the tables.

## Performance Strategy
- All numerical functions `@jax.jit` compiled
- `vmap` over objects for parallel stamp rendering
- Sort galaxies by flux; render faint/small ones as point sources (~90% of objects)
- Don't necessarily render galaxies significantly below the noise floor at all
- FFT-based PSF convolution (one pass over full image)
- Float32 throughout
- Tile-based processing to bound memory

## Verification
- Unit tests per model (Schechter sampling, mass-size, MZR)
- Integration test: render a 1'×1' field, check source counts match expectations
- Validate GSMF against Weaver+23
- Validate number counts against deep survey data
- Benchmark rendering time per image
- Determinism test: same seed + position = bit-identical output


# To Do

- ~~Add download button to web app to download image as FITS file or PNG (for RGB).~~ DONE
- ~~More options for scaling control if relevant (e.g., set the asinh parameter)~~ DONE
- ~~Double check noise thresholds and cutoffs~~ DONE — replaced hardcoded filter bandwidth/photon energy with per-filter properties (wavelength, bandwidth, QE) for all supported filters.
- ~~More graceful handling of large images and app recovery~~ DONE — added render timeout (300s), image size validation (max 20k px), MemoryError catch, async rendering in thread pool.
- ~~More extensive validation of the galaxy population properties~~ DONE — scripts/validate.py plots GSMF, mass-size, MZR, magnitude and redshift distributions vs input models
- More options for e.g. split between quiescent vs star-forming galaxies, reflect realistic color distributions, etc. This could be done by adding a "galaxy type" parameter to the galaxy model and using different SFH/photometry models for different types, with the fraction of each type as a function of mass and redshift based on observations.
- ~~Some debug plots to show the distribution of galaxy properties~~ DONE — added scripts/validate.py
- Represent some star-forming galaxies as clumpy rather than smooth Sersic profiles, to add more visual diversity and realism. This could be done by adding a "clumpiness" parameter to the galaxy model and rendering a few bright clumps in addition to the smooth component for high-SFR galaxies. Total SFR would be conserved, just distributed differently spatially. 
- ~~Pull more physical parameters into the config (e.g. mass function parameters, size relation parameters).~~ DONE
- ~~Confusion noise from unresolved faint galaxies.~~ DONE
- Optional cosmic ray, persistence, or other detector effects for more realism in certain use cases.
- ~~Debug mode to render maps colored by property (mass, size, redshift)~~ DONE — added /api/render/debug endpoint and Debug tab in UI
- Skip for now: Correlated noise model - e.g. 1/f noise for IR detectors, or spatially varying background due to scattered light. Background noise should be correlated on the scale of the PSF or larger to be realistic, rather than pure pixel-wise Gaussian noise. This could be implemented by generating a noise image with a specified power spectrum and adding it to the final image.
- ~~Add WCS to the output FITS files so that the images can be easily used in astronomy software.~~ DONE
- ~~For the API, add endpoints to query the generated catalog for a tile.~~ DONE (existed already)
- ~~Add WCS overlay to the served images (RA/Dec in cursor info).~~ DONE
- ~~Buttons to render neighboring tiles~~ DONE — added N/S/E/W pan buttons in the Pointing section that shift RA/Dec by the FoV and re-render.
- ~~Make hard-coded limits in scripts (e.g. maximum number of galaxies to render) configurable via the config system.~~ DONE
- ~~Add little projection of the sky in the app corner showing the current tile location.~~ DONE
