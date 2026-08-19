# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Unit tests for the family-neutral lazy safetensors store.
from __future__ import annotations

import pytest
import torch

from pypto_serving.model.common.weights.store import LazySafetensorsStore


class _Reader:
    """Records every name read, so a test can prove which shard a read touched."""

    def __init__(self, tensors, reads):
        self._tensors = tensors
        self._reads = reads

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_tensor(self, name):
        self._reads.append(name)
        return self._tensors[name]


def _counting_store(tmp_path, weight_map, tensors):
    """A store whose opener counts shard opens and tensor reads."""
    opens: list[str] = []
    reads: list[str] = []
    for filename in set(weight_map.values()):
        (tmp_path / filename).write_bytes(b"")

    def _open(path, device):
        opens.append(path.name)
        return _Reader(tensors, reads)

    store = LazySafetensorsStore(model_dir=tmp_path, weight_map=weight_map, safe_open_fn=_open)
    return store, opens, reads


def test_load_many_opens_each_shard_once_even_when_names_interleave(tmp_path):
    weight_map = {"a": "s0.safetensors", "b": "s1.safetensors", "c": "s0.safetensors"}
    tensors = {name: torch.tensor([i]) for i, name in enumerate(weight_map)}
    store, opens, _ = _counting_store(tmp_path, weight_map, tensors)

    store.load_many(["a", "b", "c"])

    # Grouping is the whole point: interleaved names must not reopen a shard.
    assert sorted(opens) == ["s0.safetensors", "s1.safetensors"]


def test_load_many_returns_caller_order_and_drops_duplicates(tmp_path):
    weight_map = {"a": "s0.safetensors", "b": "s0.safetensors"}
    tensors = {"a": torch.tensor([1]), "b": torch.tensor([2])}
    store, _, reads = _counting_store(tmp_path, weight_map, tensors)

    loaded = store.load_many(["b", "a", "b"])

    assert list(loaded) == ["b", "a"]
    assert reads == ["b", "a"]


def test_load_tensor_reads_only_the_requested_name(tmp_path):
    weight_map = {"wanted": "s0.safetensors", "other": "s0.safetensors"}
    tensors = {"wanted": torch.tensor([7]), "other": torch.tensor([8])}
    store, _, reads = _counting_store(tmp_path, weight_map, tensors)

    assert store.load_tensor("wanted").tolist() == [7]
    assert reads == ["wanted"]


def test_filename_for_unknown_name_raises_with_the_family_template(tmp_path):
    class _Family(LazySafetensorsStore):
        missing_name_error = "No such tensor in the Fake index: {name}"

    store = _Family(model_dir=tmp_path, weight_map={})

    with pytest.raises(KeyError, match="No such tensor in the Fake index: ghost"):
        store.filename_for("ghost")


def test_require_previews_eight_names_and_reports_the_total(tmp_path):
    store = LazySafetensorsStore(model_dir=tmp_path, weight_map={})

    with pytest.raises(KeyError) as excinfo:
        store.require([f"w{i}" for i in range(10)])

    message = str(excinfo.value)
    assert "w7" in message
    assert "w8" not in message
    assert "(10 total)" in message


def test_missing_shard_file_names_the_path(tmp_path):
    store = LazySafetensorsStore(model_dir=tmp_path, weight_map={"a": "absent.safetensors"})

    with pytest.raises(FileNotFoundError, match="absent.safetensors"):
        store.load_many(["a"])


def test_path_for_joins_the_model_dir(tmp_path):
    store = LazySafetensorsStore(model_dir=tmp_path, weight_map={"a": "s0.safetensors"})

    assert store.path_for("a") == tmp_path / "s0.safetensors"


def test_contains_rejects_non_strings(tmp_path):
    store = LazySafetensorsStore(model_dir=tmp_path, weight_map={"a": "s0.safetensors"})

    assert "a" in store
    assert "b" not in store
    assert 1 not in store


def test_reads_a_real_safetensors_shard_by_name(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {"kept": torch.arange(4, dtype=torch.float32).reshape(2, 2), "ignored": torch.ones(2, 2)},
        str(tmp_path / "shard.safetensors"),
    )
    store = LazySafetensorsStore(
        model_dir=tmp_path,
        weight_map={"kept": "shard.safetensors", "ignored": "shard.safetensors"},
    )

    assert store.load_tensor("kept").tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_family_default_opener_is_resolved_per_construction(tmp_path):
    """A family points at its own module-level opener, and it is looked up per instance.

    The DeepSeek store relies on this: its opener carries a family-specific import
    diagnostic, and tests replace that module attribute before constructing a store.
    """
    calls: list[str] = []

    class _Family(LazySafetensorsStore):
        def _default_open_fn(self):
            def _open(path, device):
                calls.append(path.name)
                return _Reader({"a": torch.tensor([1])}, [])

            return _open

    (tmp_path / "s0.safetensors").write_bytes(b"")
    store = _Family(model_dir=tmp_path, weight_map={"a": "s0.safetensors"})
    store.load_tensor("a")

    assert calls == ["s0.safetensors"]
