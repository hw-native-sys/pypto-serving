# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check read-only diagnostics against permuted page bytes and failed receipts."""

import copy
import ctypes
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[4] / ".agents/skills/profile-dsv4-lowlat-strace/scripts"
sys.path.insert(0, str(SCRIPTS))
from lowlat_prefix_audit import cache_hashes, tensor_hash, validate
from lowlat_profile import server_command
from pypto_serving.model.deepseek_lowlat.cache import TENSOR_GROUPS


def test_history_hash_preserves_bf16_storage_bits():
    bits = torch.tensor([0, -32768, 1, 16256, 32704], dtype=torch.int16)
    assert tensor_hash(bits.view(torch.bfloat16)) == hashlib.sha256(bits.numpy().tobytes()).hexdigest()
    assert tensor_hash(bits.view(torch.bfloat16)[::2]) == hashlib.sha256(bits[::2].numpy().tobytes()).hexdigest()


def test_hashes_follow_logical_pages_and_borrowed_offsets_without_writing():
    roots = {}
    shards = []
    expected = []
    original = []
    for rank in range(8):
        values = torch.arange(3 * 4 * 8, dtype=torch.int16).reshape(3, 4, 8) + rank * 100
        payload = values.view(torch.uint8).reshape(-1)
        root = torch.cat((torch.full((16,), 237, dtype=torch.uint8), payload, torch.full((16,), 239, dtype=torch.uint8)))
        base = root.data_ptr()
        roots[base] = root
        original.append(root.clone())
        shards.append(SimpleNamespace(shape=(3, 4, 8), dtype=torch.int16,
                                      data_ptr=base+16, byte_offset=16))
        expected.append(hashlib.sha256(values[[2, 0]].numpy().tobytes()).hexdigest())

    class Worker:
        def copy_from(self, destination, source, nbytes, *, src_offset, worker_id):
            assert source in roots
            root = roots[source]
            assert root is list(roots.values())[worker_id]
            ctypes.memmove(destination, source + src_offset, nbytes)

    owner = SimpleNamespace(shape=(8, 3, 4, 8), shards=shards, dtype=torch.int16)
    cache = SimpleNamespace(owners={"kv_cache": owner}, specs={"ori": SimpleNamespace(num_blocks=3)},
                            arena=SimpleNamespace(worker=Worker()))
    interval = SimpleNamespace(group="ori", start=0, end=2)
    scratch = torch.empty(17, dtype=torch.uint8).share_memory_()
    assert cache_hashes(cache, {"ori": (2, 0, 1)}, [interval], scratch) == {"kv_cache": expected}
    assert all(torch.equal(before, after) for before, after in zip(original, roots.values()))
    # Page1 is outside the proof; page2 is part of the declared live state.
    root = list(roots.values())[0]
    root[16 + 64] ^= 1
    assert cache_hashes(cache, {"ori": (2, 0, 1)}, [interval], scratch)["kv_cache"] == expected
    root[16 + 128] ^= 1
    assert cache_hashes(cache, {"ori": (2, 0, 1)}, [interval], scratch)["kv_cache"][0] != expected[0]
    with pytest.raises(ValueError, match="distinct"):
        cache_hashes(cache, {"ori": (0, 0)}, [interval], scratch)


def records():
    common = dict(prompt_tokens=1536, output_tokens=2, token_ids=[17, 29], complete=True,
                  prefill_cache={name: ["same"]*8 for name in TENSOR_GROUPS}, history_seed="history",
                  device_history=["history", *(["zero"]*7)], native_seed=[[1, 1535, 1, 17]]*8)
    result = []
    for index, kind in enumerate(("checkpoint", "restore", "restore")):
        row = copy.deepcopy(common)
        row.update(request_id=str(index), copies=[dict(kind=kind, hashes={name: ["checkpoint"]*8 for name in TENSOR_GROUPS},
                                                     ranges=[dict(group="ori", start=10, end=11)])])
        row["native_seed"] = [[index*10+1, 1535, 1, 17] for _ in range(8)]
        result.append(row)
    return result


@pytest.mark.parametrize("damage", [None, "tokens", "rank", "seed", "checkpoint", "incomplete", "missing", "partial"])
def test_receipt_rejects_changed_state_or_incomplete_evidence(tmp_path, damage):
    rows = records()
    if damage == "tokens": rows[1]["token_ids"][1] = 30
    if damage == "rank": rows[1]["prefill_cache"]["kv_cache"][7] = "corrupt"
    if damage == "seed": rows[2]["native_seed"][3][0] += 1
    if damage == "checkpoint": rows[2]["copies"][0]["hashes"]["kv_cache"][0] = "modified"
    if damage == "incomplete": rows[2]["complete"] = False
    if damage == "missing": rows.pop()
    if damage == "partial":
        for row in rows: row["prefill_cache"]["kv_cache"].pop()
    for row in rows: (tmp_path/(row["request_id"]+".json")).write_text(json.dumps(row))
    if damage is None:
        assert validate(tmp_path)["valid"] is True
        assert validate(tmp_path)["requests"] == 3
    else:
        with pytest.raises(ValueError): validate(tmp_path)
        assert not (tmp_path/"validation.json").exists()


def test_diagnostics_select_a_skill_entry_without_changing_the_product_cli():
    args = SimpleNamespace(audit_prefix_state=True, model_dir=Path("/model"), artifact_dir=Path("/artifact"),
                           max_model_len=16384, prefill_chunk_size=512, npu_memory_utilization=0.93,
                           prefix_cache_mode="hot", ring_heap="1,2,3,4")
    command = server_command(args, list(range(8)), 1234)
    assert command[1].endswith("/lowlat_prefix_audit.py")
    assert "--audit-prefix-state" not in command
    args.audit_prefix_state=False
    assert server_command(args,list(range(8)),1234)[1:3] == ["-m", "pypto_serving.cli"]
