"""Tests for hierarchical deterministic seeding."""

import jax
import jax.numpy as jnp

from skysim.seed import (
    full_key,
    layer_key,
    master_key,
    object_key,
    object_keys,
    tile_key,
)


def _keys_equal(a, b):
    return bool(jnp.all(a == b))


def test_master_key_deterministic():
    """Same seed produces the same key."""
    assert _keys_equal(master_key(42), master_key(42))


def test_master_key_different_seeds():
    """Different seeds produce different keys."""
    assert not _keys_equal(master_key(0), master_key(1))


def test_tile_key_deterministic():
    mk = master_key(99)
    assert _keys_equal(tile_key(mk, 100), tile_key(mk, 100))


def test_tile_key_varies_with_index():
    mk = master_key(99)
    assert not _keys_equal(tile_key(mk, 0), tile_key(mk, 1))


def test_layer_key_deterministic():
    tk = tile_key(master_key(0), 10)
    assert _keys_equal(layer_key(tk, "galaxies"), layer_key(tk, "galaxies"))


def test_layer_key_varies_with_name():
    tk = tile_key(master_key(0), 10)
    assert not _keys_equal(layer_key(tk, "galaxies"), layer_key(tk, "stars"))


def test_object_key_deterministic():
    lk = layer_key(tile_key(master_key(0), 5), "galaxies")
    assert _keys_equal(object_key(lk, 0), object_key(lk, 0))


def test_object_key_varies():
    lk = layer_key(tile_key(master_key(0), 5), "galaxies")
    assert not _keys_equal(object_key(lk, 0), object_key(lk, 1))


def test_object_keys_shape():
    lk = layer_key(tile_key(master_key(0), 5), "galaxies")
    keys = object_keys(lk, 10)
    assert keys.shape == (10, 2)


def test_full_key_matches_manual():
    """full_key convenience function matches step-by-step construction."""
    seed, tile_idx, layer, obj = 42, 7, "galaxies", 3
    manual = object_key(layer_key(tile_key(master_key(seed), tile_idx), layer), obj)
    auto = full_key(seed, tile_idx, layer, obj)
    assert _keys_equal(manual, auto)


def test_full_key_without_object():
    seed, tile_idx, layer = 42, 7, "galaxies"
    manual = layer_key(tile_key(master_key(seed), tile_idx), layer)
    auto = full_key(seed, tile_idx, layer)
    assert _keys_equal(manual, auto)


def test_hierarchy_independence():
    """Changing tile index doesn't affect a different tile's keys."""
    k1 = full_key(0, 10, "galaxies", 5)
    k2 = full_key(0, 11, "galaxies", 5)
    assert not _keys_equal(k1, k2)


def test_draws_are_deterministic():
    """Actual random draws from the same key are identical."""
    k = full_key(0, 0, "galaxies")
    a = jax.random.normal(k, shape=(100,))
    b = jax.random.normal(k, shape=(100,))
    assert jnp.allclose(a, b)
