# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical-table equivalence and slot/lease isolation without device execution."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pypto_serving.config.types import DecodeBatch
from pypto_serving.model.deepseek_dspark import npu_runner as mod
from pypto_serving.model.deepseek_dspark import task_args
from pypto_serving.model.deepseek_dspark.cache_tables import DecodeTableRows, DrafterTableRows
from tests.unit.model.deepseek_dspark.test_dspark_model import _block_rows, _runner


@pytest.mark.parametrize("ring", [False, True])
def test_changed_row_only_updates_its_replicas_and_owner(ring):
    table = torch.empty((4, 17), dtype=torch.int32)
    replica = torch.empty_like(table)
    owner = torch.empty((2, 17), dtype=torch.int32)
    cache = DecodeTableRows(table, ring=ring, fillers=range(20, 24), replicas=[replica], owner=owner)
    before = table.clone()
    assert not cache.update(0, None, (20,))
    assert cache.update(1, ("a", 1), (3, 7, 9))
    np.testing.assert_array_equal(table[1].numpy(),
        np.resize([3, 7, 9], 17) if ring else [3, 7, 9] + [-1] * 14)
    torch.testing.assert_close(table, replica)
    torch.testing.assert_close(table[:2], owner)
    torch.testing.assert_close(table[[0, 2, 3]], before[[0, 2, 3]])
    assert not cache.update(1, ("a", 1), (3, 7, 9))
    assert cache.update(1, ("a", 2), (3, 7, 9))
    assert cache.update(1, ("b", 2), (4,))
    assert cache.update(1, None, (21,))
    torch.testing.assert_close(table, before)
    torch.testing.assert_close(table, replica)
    torch.testing.assert_close(table[:2], owner)


@pytest.mark.parametrize("blocks", [(), (-1,), tuple(range(18))])
def test_invalid_absolute_row_does_not_replace_its_binding(blocks):
    table = torch.empty((1, 17), dtype=torch.int32)
    cache = DecodeTableRows(table, ring=False, fillers=[20])
    before = table.clone()
    with pytest.raises(ValueError, match="invalid physical block row"):
        cache.update(0, ("a", 1), blocks)
    torch.testing.assert_close(table, before)
    assert not cache.update(0, None, (20,))


_FIELDS = {"ori": "ori", "cmp_c128": "hca_cmp", "cmp_c4": "csa_cmp", "idx": "idx",
           "hca_state": "hca_state", "csa_state": "csa_state", "csa_inner_state": "csa_inner_state"}
_OWNER = {"cmp_c128": "hca_cmp_block_table", "cmp_c4": "csa_cmp_block_table", "idx": "csa_idx_block_table"}


def _fused_runner(tp):
    runner = _runner(speculative=True, tp_size=tp)
    runner._compiled.decode_full_fused = True
    layout = runner._compiled.layout
    runner._cache_group_specs = mod.build_dspark_cache_group_specs(3, (0, 4, 128), max_prefill_tokens=128,
                                                                partitions=layout.partitions)
    runner._cache_group_num_blocks = {name: 32768 for name in _FIELDS}
    second = task_args.decode_task_args(runner)
    second.allocate_host_shared(None)
    runner._decode_task_args.append(second)
    depths = {"ori": mod.DSPARK_DECODE_ORI_TABLE_BLOCKS,
              "cmp_c128": mod.DSPARK_DECODE_HCA_CMP_TABLE_BLOCKS,
              "cmp_c4": mod.DSPARK_DECODE_CMP_C4_TABLE_BLOCKS, "idx": mod.DSPARK_DECODE_IDX_TABLE_BLOCKS,
              **{name: runner._decode_state_table_depth(name) for name in
                 ("hca_state", "csa_state", "csa_inner_state")}}
    for _ in range(2):
        buffers = {f"group_{field}_block_tables": torch.empty((16, 64, depths[name]), dtype=torch.int32)
                   for name, field in _FIELDS.items()}
        buffers.update({name: torch.empty((16, rows), dtype=torch.int32) for name, rows in (
            ("state_slot_ids", layout.decode_local_batch), ("state_generations", layout.decode_local_batch),
            ("group_state_slot_ids", 64), ("group_state_generations", 64))})
        runner._dspark_state_buffers.append(SimpleNamespace(**buffers))
    runner._fused_drafter_block_tables = [{4: torch.empty((16, 3, 4, mod.DSPARK_DRAFTER_TABLE_BLOCKS),
                                                       dtype=torch.int32)} for _ in range(2)]
    for tables in runner._fused_drafter_block_tables:
        for table in tables.values():
            runner._drafter_table_rows[id(table)] = DrafterTableRows(
                table, runner._drafter_layer_patterns, mod.DSPARK_DRAFTER_FILLER_BLOCK_BASE,
            )
    runner._bind_fused_decode_args = lambda _: ()
    runner._initialize_decode_table_rows()
    return runner


def _check_full_reference(runner, plan):
    """The deliberately full algorithm is a test oracle, not a runtime mode."""
    layout = runner._compiled.layout
    scratch = runner._scratch_blocks()
    for group, group_plan in enumerate(plan.group_plans):
        begin, end = group * layout.tp_size, (group + 1) * layout.tp_size
        for name, field in _FIELDS.items():
            tables = getattr(plan, f"group_{field}_block_tables")
            depth = tables.shape[-1]
            expected = np.full((64, depth), -1, dtype=np.int32)
            for row, index in enumerate(group_plan.request_indices):
                ids = (scratch[name][row],) if index is None else plan.request_blocks[index][name]
                if name not in _OWNER:
                    expected[row] = np.resize(ids, depth)
                else:
                    expected[row, :len(ids)] = ids
            for rank in range(begin, end):
                np.testing.assert_array_equal(tables[rank].numpy(), expected)
                if name in _OWNER:
                    offset = (rank - begin) * layout.decode_local_batch
                    owner = runner._decode_task_args[plan.buffer_slot].tensors[_OWNER[name]][rank]
                    np.testing.assert_array_equal(owner.numpy(), expected[offset:offset + layout.decode_local_batch])


@pytest.mark.parametrize("tp", [1, 4])
def test_fused_tables_match_full_geometry_across_slots_cow_rebind_and_exit(monkeypatch, tp):
    runner = _fused_runner(tp)
    last_group = runner._compiled.layout.partitions - 1
    for request, group, lease in (("a", 0, 0), ("b", 0, tp), ("c", last_group, 0)):
        runner._drafter_states[request] = SimpleNamespace(group=group, lease=lease, generation=1)
    batch = DecodeBatch(request_ids=["a", "b", "c"], token_ids=torch.ones((3, 1), dtype=torch.int64),
                        hidden_states=None, seq_lens=torch.tensor([31, 127, 128], dtype=torch.int32),
                        block_ids_by_group=_block_rows(3), cache_partitions=[0, 0, last_group])
    changed = []
    original = DecodeTableRows.update
    def record(self, row, identity, blocks):
        result = original(self, row, identity, blocks)
        if result:
            changed.append((row, identity, blocks))
        return result
    monkeypatch.setattr(DecodeTableRows, "update", record)
    first = runner._prepare_decode_plan(batch, buffer_slot=0)
    runner._stage_device_prepared_decode(first)
    _check_full_reference(runner, first)
    assert len(changed) == 3 * 7
    saved = {name: getattr(first, f"group_{field}_block_tables").clone() for name, field in _FIELDS.items()}
    pending = object()
    runner._track_pending_decode_dispatch(0, pending)
    with pytest.raises(RuntimeError, match="dispatch is pending"):
        runner._prepare_decode_plan(batch, buffer_slot=0)
    second = runner._prepare_decode_plan(batch, buffer_slot=1)
    runner._stage_device_prepared_decode(second)
    _check_full_reference(runner, second)
    for name, field in _FIELDS.items():
        torch.testing.assert_close(getattr(first, f"group_{field}_block_tables"), saved[name])
    runner._forget_pending_decode_dispatch(0, pending)
    # Positions cross 32/128 but the physical snapshot has not changed.
    changed.clear()
    batch.seq_lens += 8
    runner._stage_device_prepared_decode(runner._prepare_decode_plan(batch, buffer_slot=0))
    assert changed == []
    # A state COW changes only its own group, not compressed history.
    for name in ("hca_state", "csa_state", "csa_inner_state"):
        changed.clear()
        ids = list(range(501, 501 + len(batch.block_ids_by_group[0][name])))
        batch.block_ids_by_group[0][name] = ids
        _check_full_reference(runner, runner._prepare_decode_plan(batch, buffer_slot=0))
        assert changed == [(0, ("a", 1), tuple(ids))]
    # Compressed-page append/shrink clears the old tail in just that row.
    for name in _OWNER:
        for ids in ([21, 22, 23], [21]):
            changed.clear()
            batch.block_ids_by_group[0][name] = ids
            _check_full_reference(runner, runner._prepare_decode_plan(batch, buffer_slot=0))
            assert changed == [(0, ("a", 1), tuple(ids))]
    # Same lease and physical IDs, new generation: all groups must rebind.
    changed.clear()
    runner._drafter_states["a"].generation += 1
    _check_full_reference(runner, runner._prepare_decode_plan(batch, buffer_slot=0))
    assert len(changed) == 7
    # A new request can reuse the exact same physical lease/mapping safely.
    runner._drafter_states["replacement"] = runner._drafter_states.pop("a")
    batch.request_ids[0] = "replacement"
    changed.clear()
    _check_full_reference(runner, runner._prepare_decode_plan(batch, buffer_slot=0))
    assert len(changed) == 7
    # Dense owner rows swap, then disappear; old tails must become fillers.
    for indices in ((1, 0, 2), (2,)):
        moved = replace(batch, request_ids=[batch.request_ids[i] for i in indices],
                        block_ids_by_group=[batch.block_ids_by_group[i] for i in indices],
                        cache_partitions=[batch.cache_partitions[i] for i in indices],
                        token_ids=batch.token_ids[list(indices)], seq_lens=batch.seq_lens[list(indices)])
        plan = runner._prepare_decode_plan(moved, buffer_slot=0)
        runner._stage_device_prepared_decode(plan)
        _check_full_reference(runner, plan)
    # Slot one still holds its old, independently valid mapping.
    _check_full_reference(runner, second)
    runner.close()


@pytest.mark.parametrize("tp", [1, 4])
def test_drafter_reuses_each_slot_bucket_and_rebinds_lease_generation(monkeypatch, tp):
    runner = _runner(speculative=True, tp_size=tp)
    slots = [{b: torch.empty((16, 3, b, mod.DSPARK_DRAFTER_TABLE_BLOCKS), dtype=torch.int32)
              for b in (4, 8)} for _ in range(2)]
    for tables in slots:
        for table in tables.values():
            runner._drafter_table_rows[id(table)] = DrafterTableRows(
                table, runner._drafter_layer_patterns, mod.DSPARK_DRAFTER_FILLER_BLOCK_BASE,
            )
    patterns = np.stack([(np.arange(mod.DSPARK_DRAFTER_TABLE_BLOCKS) + 7 * layer)
                         % mod.DSPARK_DRAFTER_RING_BLOCKS for layer in range(3)])
    changed = []
    original = DrafterTableRows.update
    def record(self, rank, row, identity, base):
        result = original(self, rank, row, identity, base)
        if result:
            changed.append((rank, row, identity))
        return result
    monkeypatch.setattr(DrafterTableRows, "update", record)
    def check(slot, count, bucket, anchor=31, order=None):
        leases = list(range(count)) if order is None else list(order)
        rows = [[] for _ in range(16)]
        rows[0] = [mod.DSparkDrafterRequestRow(request_id=f"r{i}", group=0, lease=i,
                    anchor=anchor, valid_count=8, token_source=1, hidden_row=row * 8, decode_mode=True)
                   for row, i in enumerate(leases)]
        runner._drafter_block_tables(rows, bucket, staging=slots[slot])
        bases = np.full((16, bucket), mod.DSPARK_DRAFTER_FILLER_BLOCK_BASE, dtype=np.int32)
        bases[0, :count] = np.array(leases, dtype=np.int32) * mod.DSPARK_DRAFTER_RING_BLOCKS
        expected = bases[:, None, :, None] + patterns[None, :, None, :]
        np.testing.assert_array_equal(slots[slot][bucket].numpy(), expected)
    check(0, 2, 4)
    saved = slots[0][4].clone()
    changed.clear()
    check(1, 2, 4)
    assert len(changed) == 2
    changed.clear()
    check(0, 2, 4, anchor=135)
    assert changed == []
    check(0, 5, 8)
    assert len(changed) == 5
    changed.clear()
    check(0, 2, 4)
    assert changed == []
    check(0, 2, 4, order=(1, 0))
    assert len(changed) == 2
    changed.clear()
    check(0, 2, 4)
    assert len(changed) == 2
    changed.clear()
    runner._drafter_lease_generations[0][0] += 1
    check(0, 2, 4)
    assert len(changed) == 1
    changed.clear()
    check(0, 1, 4)
    assert changed == [(0, 1, None)]
    torch.testing.assert_close(slots[1][4], saved)
    check(0, 0, 4)
    runner.close()
