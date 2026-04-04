"""Hierarchical deterministic seeding using JAX PRNG.

Key hierarchy:
    master_seed
    └─ tile (fold_in tile index)
       └─ layer (fold_in layer id)
          └─ object (fold_in object index)

This ensures any sky tile is reproducible independently.
"""

from __future__ import annotations

import hashlib
from typing import Union

import jax
import jax.numpy as jnp


def master_key(seed: int) -> jax.Array:
    """Create the root PRNG key from an integer seed."""
    return jax.random.PRNGKey(seed)


def tile_key(master: jax.Array, tile_index: int) -> jax.Array:
    """Derive a per-tile key by folding in the tile index."""
    return jax.random.fold_in(master, tile_index)


def layer_key(tile: jax.Array, layer_name: str) -> jax.Array:
    """Derive a per-layer key by folding in a hash of the layer name.

    Using a hash lets us use readable string names while still
    producing a deterministic integer for fold_in.
    """
    h = hashlib.sha256(layer_name.encode()).digest()
    layer_id = int.from_bytes(h[:4], "big")
    return jax.random.fold_in(tile, layer_id)


def object_key(parent: jax.Array, object_index: Union[int, jax.Array]) -> jax.Array:
    """Derive a per-object key by folding in the object index."""
    return jax.random.fold_in(parent, object_index)


def object_keys(parent: jax.Array, n: int) -> jax.Array:
    """Split a parent key into *n* independent sub-keys (one per object).

    Returns an array of shape (n, 2) — one key per object.
    This is useful when you want to vmap over objects.
    """
    return jax.random.split(parent, n)


def full_key(
    seed: int,
    tile_index: int,
    layer_name: str,
    object_index: Union[int, jax.Array, None] = None,
) -> jax.Array:
    """Convenience: derive a key for any level of the hierarchy in one call."""
    key = master_key(seed)
    key = tile_key(key, tile_index)
    key = layer_key(key, layer_name)
    if object_index is not None:
        key = object_key(key, object_index)
    return key
