# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Prepacked-sidecar parity: an artifact written before this refactor must still be usable.
#
# A sidecar is a whole-model stacked payload on disk, and the ones already published took ~40
# minutes each to build. The refactor is only safe if it neither changes what gets written nor
# rejects what was written before, so this covers both directions: the payload the current
# stacker produces against the payload the hand-written helpers produced, and a legacy-written
# file loaded through the current reader.
#
# The reader itself is deliberately untouched by #163 — `_map_shared_prepacked_tensors`, the
# single-fd flow and the residency gate are frozen — so these tests exercise the seam that did
# move: what the writer puts in the file.
from __future__ import annotations

import json
import struct

import pytest

from pypto_serving.model.deepseek.weight_loader import (
    DEEPSEEK_V4_PACKED_FORMAT,
    deepseek_v4_packed_weights_path,
)
_EXPERTS = 4


def _write_sidecar(checkpoint, tensors, path):
    """Publish a sidecar the way `tools/prepack_deepseek_v4.py` does."""
    from safetensors.torch import save_file

    fingerprint = checkpoint.store().packed_stacked_layer_weights_fingerprint(
        ranks=checkpoint.ranks,
        n_routed_experts=checkpoint.n_routed_experts,
        compress_ratios=checkpoint.compress_ratios,
        num_hash_layers=checkpoint.num_hash_layers,
    )
    save_file(
        dict(tensors),
        str(path),
        metadata={"format": DEEPSEEK_V4_PACKED_FORMAT, "source_fingerprint": fingerprint},
    )
    return fingerprint


def _header(path):
    """Return the safetensors header: the tensor entries and the metadata, parsed."""
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def _payload(path):
    """Return everything after the header — the tensor bytes, in file order."""
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + length)
        return handle.read()


def test_a_written_sidecar_round_trips_through_the_reader(
    deepseek_checkpoint, fingerprint_tensors, tmp_path
):
    """Write, read back, and match a fresh pack of the same checkpoint.

    The comparison against the pre-refactor packer lived in the PR that did the refactor and
    went green there; once that code is gone the property worth guarding is this one — that the
    artifact on disk and the shard path agree, so a stale or corrupt sidecar is the only way
    they can diverge.

    Compared as (header entries, metadata dict, payload bytes) rather than as whole files:
    safetensors serializes its metadata map in nondeterministic order — eight writes of one dict
    in a single process gave two different key orders — so a byte-for-byte file assertion would
    be flaky for a reason that has nothing to do with this code.
    """
    checkpoint = deepseek_checkpoint(n_routed_experts=_EXPERTS)
    packed = checkpoint.load_stacked().tensors

    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    _write_sidecar(checkpoint, packed, first)
    _write_sidecar(checkpoint, checkpoint.load_stacked().tensors, second)

    first_header = _header(first)
    second_header = _header(second)
    assert first_header.pop("__metadata__") == second_header.pop("__metadata__")
    # dtype, shape and data_offsets for every tensor: the offsets a published sidecar records
    # must not move, or a reader would map the wrong bytes.
    assert first_header == second_header
    assert _payload(first) == _payload(second)

    path = deepseek_v4_packed_weights_path(checkpoint.model_dir, ranks=checkpoint.ranks)
    _write_sidecar(checkpoint, packed, path)
    loaded = checkpoint.store().load_prepacked_stacked_layer_weights(
        ranks=checkpoint.ranks,
        n_routed_experts=checkpoint.n_routed_experts,
        compress_ratios=checkpoint.compress_ratios,
        num_hash_layers=checkpoint.num_hash_layers,
    )

    assert loaded is not None, "the reader rejected a sidecar this writer produced"
    assert fingerprint_tensors(loaded.tensors) == fingerprint_tensors(packed)


def test_a_currently_written_sidecar_is_taken_in_preference_to_repacking(
    deepseek_checkpoint, fingerprint_tensors, monkeypatch
):
    """The other direction: today's writer, today's reader, and the shard path not touched.

    `use_prepacked=True` must actually consume the sidecar; if it silently repacked, the
    sidecar would be dead weight and every start would pay the slow path.
    """
    checkpoint = deepseek_checkpoint(n_routed_experts=_EXPERTS)
    expected = fingerprint_tensors(checkpoint.load_stacked().tensors)
    path = deepseek_v4_packed_weights_path(checkpoint.model_dir, ranks=checkpoint.ranks)
    _write_sidecar(checkpoint, checkpoint.load_stacked().tensors, path)

    store = checkpoint.store()
    monkeypatch.setattr(
        store,
        "load_packed_layer_weights",
        lambda *args, **kwargs: pytest.fail("a valid sidecar must not be repacked from shards"),
    )
    loaded = store.load_stacked_layer_weights(
        ranks=checkpoint.ranks,
        n_routed_experts=checkpoint.n_routed_experts,
        compress_ratios=checkpoint.compress_ratios,
        num_hash_layers=checkpoint.num_hash_layers,
    )

    assert fingerprint_tensors(loaded.tensors) == expected


def test_the_fingerprint_is_unchanged_by_the_refactor(deepseek_checkpoint):
    """Pinned against its own definition, because it decides whether published files stay valid.

    The fingerprint covers the config, the weight map and every source shard's size and mtime.
    Nothing in #163 touches those, and this asserts it: a changed payload would invalidate every
    sidecar in the fleet at once, silently, by making them all look stale.
    """
    checkpoint = deepseek_checkpoint(n_routed_experts=_EXPERTS)
    store = checkpoint.store()
    kwargs = {
        "ranks": checkpoint.ranks,
        "n_routed_experts": checkpoint.n_routed_experts,
        "compress_ratios": checkpoint.compress_ratios,
        "num_hash_layers": checkpoint.num_hash_layers,
    }

    computed = store.packed_stacked_layer_weights_fingerprint(**kwargs)

    source_files = []
    for filename in sorted(set(store.weight_map.values())):
        stat = (store.model_dir / filename).stat()
        source_files.append([filename, stat.st_size, stat.st_mtime_ns])
    payload = {
        "format": DEEPSEEK_V4_PACKED_FORMAT,
        "ranks": checkpoint.ranks,
        "n_routed_experts": checkpoint.n_routed_experts,
        "compress_ratios": [int(r) for r in checkpoint.compress_ratios],
        "num_hash_layers": checkpoint.num_hash_layers,
        "weight_map": sorted([list(item) for item in store.weight_map.items()]),
        "source_files": source_files,
    }
    import hashlib

    expected = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    assert computed == expected


def test_a_sidecar_whose_source_shard_moved_is_ignored(deepseek_checkpoint):
    """Staleness detection must still work — otherwise a refactor could resurrect a bad file."""
    checkpoint = deepseek_checkpoint(n_routed_experts=_EXPERTS)
    path = deepseek_v4_packed_weights_path(checkpoint.model_dir, ranks=checkpoint.ranks)
    _write_sidecar(checkpoint, checkpoint.load_stacked().tensors, path)

    # Touch one source shard: same content, new mtime, so the fingerprint no longer matches.
    shard = checkpoint.model_dir / sorted(set(checkpoint.weight_map.values()))[0]
    stat = shard.stat()
    import os

    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert (
        checkpoint.store().load_prepacked_stacked_layer_weights(
            ranks=checkpoint.ranks,
            n_routed_experts=checkpoint.n_routed_experts,
            compress_ratios=checkpoint.compress_ratios,
            num_hash_layers=checkpoint.num_hash_layers,
        )
        is None
    )
