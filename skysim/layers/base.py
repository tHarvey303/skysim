"""Layer protocol — the interface every sky layer must implement."""

from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from skysim.config import SimConfig
from skysim.coordinates import TileInfo

# A catalog is a struct-of-arrays: column name → 1-D JAX array.
Catalog = Dict[str, jnp.ndarray]


@runtime_checkable
class Layer(Protocol):
    """Protocol that all sky layers (galaxies, stars, LSS, …) must satisfy."""

    name: str

    def generate_catalog(
        self,
        key: jax.Array,
        tile: TileInfo,
        config: SimConfig,
    ) -> Catalog:
        """Sample a catalog of sources for a single tile.

        Parameters
        ----------
        key : jax.Array
            PRNG key already scoped to this tile+layer.
        tile : TileInfo
            Metadata about the sky tile being generated.
        config : SimConfig
            Global simulation configuration.

        Returns
        -------
        Catalog
            Dict mapping column names to 1-D arrays, e.g.
            {"ra": …, "dec": …, "flux": …, "half_light_radius": …}.
        """
        ...

    def render(
        self,
        catalog: Catalog,
        image: jnp.ndarray,
        config: SimConfig,
    ) -> jnp.ndarray:
        """Render cataloged sources onto an image.

        Parameters
        ----------
        catalog : Catalog
            Output of generate_catalog().
        image : jnp.ndarray
            2-D float32 image to add sources into (modified in-place semantically;
            JAX arrays are immutable so a new array is returned).
        config : SimConfig
            Global simulation configuration.

        Returns
        -------
        jnp.ndarray
            Updated image with this layer's sources added.
        """
        ...
