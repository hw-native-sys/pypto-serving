# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Unit tests for the whole-model weight rules and the staging pipeline.
from __future__ import annotations

import threading

import pytest
import torch

from pypto_serving.model.common.weights.packer import pack_globals, pad_rows
from pypto_serving.model.common.weights.pipeline import (
    StagingPolicy,
    stage_and_release,
    stage_layers,
)
from pypto_serving.model.common.weights.spec import GlobalWeightRule
from pypto_serving.model.common.weights.shard import TensorParallel


class TestGlobalWeightRules:
    """Resolution, the tied-weight fallback, and the two padding fills."""

    def test_a_tied_checkpoint_falls_back_to_the_embedding(self):
        """No `lm_head` in the file means tied weights, not a missing weight."""
        embed = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        rule = GlobalWeightRule("lm_head", "lm_head.weight", torch.float32, fallback_source="embed.weight")

        packed = pack_globals([rule], {"embed.weight": embed})

        assert torch.equal(packed["lm_head"], embed)

    def test_the_fallback_is_only_used_when_the_source_is_absent(self):
        """An untied checkpoint must keep its own head, not silently prefer the embedding."""
        embed = torch.zeros(3, 2)
        head = torch.ones(3, 2)
        rule = GlobalWeightRule("lm_head", "lm_head.weight", torch.float32, fallback_source="embed.weight")

        packed = pack_globals([rule], {"embed.weight": embed, "lm_head.weight": head})

        assert torch.equal(packed["lm_head"], head)

    def test_a_weight_absent_with_no_fallback_names_both_candidates(self):
        rule = GlobalWeightRule("lm_head", "lm_head.weight", torch.float32, fallback_source="embed.weight")

        with pytest.raises(KeyError, match="lm_head.weight.*embed.weight"):
            pack_globals([rule], {})

    def test_an_embedding_pads_with_zeros(self):
        """A padded token id is never looked up, so zeros are the neutral choice."""
        embed = torch.ones(3, 2)
        rule = GlobalWeightRule("embed", "embed.weight", torch.float32, pad_to_multiple=4)

        packed = pack_globals([rule], {"embed.weight": embed})

        assert packed["embed"].shape == (4, 2)
        assert torch.all(packed["embed"][3] == 0.0)

    def test_an_lm_head_pads_by_replicating_its_first_row(self):
        """Not zeros, and this is the case a wrong choice would survive review.

        Zero rows in an LM head give every padded vocabulary entry the same finite logit rather
        than an impossible one, so the mistake shows up as sampling noise instead of a crash.
        """
        head = torch.stack([torch.full((2,), 5.0), torch.full((2,), 7.0), torch.full((2,), 9.0)])
        rule = GlobalWeightRule(
            "lm_head",
            "lm_head.weight",
            torch.float32,
            pad_to_multiple=4,
            pad_fill="first_row",
        )

        packed = pack_globals([rule], {"lm_head.weight": head})

        assert packed["lm_head"].shape == (4, 2)
        assert torch.all(packed["lm_head"][3] == 5.0), "the padded row must copy row 0"

    def test_an_explicit_row_target_overrides_the_multiple(self):
        """A fused kernel hard-codes its padded vocabulary; the caller states it, not the rule."""
        embed = torch.ones(3, 2)
        rule = GlobalWeightRule("embed", "embed.weight", torch.float32, pad_to_multiple=4)

        packed = pack_globals([rule], {"embed.weight": embed}, padded_rows={"embed": 8})

        assert packed["embed"].shape == (8, 2)

    def test_a_weight_that_already_fits_is_not_copied_for_padding(self):
        embed = torch.ones(4, 2)
        rule = GlobalWeightRule("embed", "embed.weight", torch.float32, pad_to_multiple=4)

        packed = pack_globals([rule], {"embed.weight": embed})

        assert packed["embed"].shape == (4, 2)

    def test_padding_down_is_refused(self):
        """Truncating a vocabulary would drop real tokens; it can only be a caller error."""
        with pytest.raises(ValueError, match="cannot pad down"):
            pad_rows("embed", torch.ones(8, 2), rows=4, fill="zeros")

    def test_an_unknown_fill_is_refused(self):
        with pytest.raises(ValueError, match="unsupported pad fill"):
            pad_rows("embed", torch.ones(2, 2), rows=4, fill="mean")

    def test_flatten_to_row_and_dtype_are_applied(self):
        rule = GlobalWeightRule("norm", "norm.weight", torch.float32, flatten_to_row=True)

        packed = pack_globals([rule], {"norm.weight": torch.ones(4, dtype=torch.bfloat16)})

        assert packed["norm"].shape == (1, 4)
        assert packed["norm"].dtype == torch.float32
        assert packed["norm"].is_contiguous()


class TestTensorParallelPolicy:
    """TP shards repeat across DP groups and preserve destination parity."""

    def test_shards_one_axis_and_repeats_it_across_dp_groups(self):
        source = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        policy = TensorParallel(ranks=8, tp_size=4, axis=1)

        packed = policy.apply("weight", source, dtype=torch.bfloat16, destination=None)

        assert packed.shape == (8, 3, 2)
        for rank in range(8):
            expected = source[:, (rank % 4) * 2 : (rank % 4 + 1) * 2].to(torch.bfloat16)
            assert torch.equal(packed[rank], expected)

    def test_preallocated_destination_matches_direct_pack(self):
        source = torch.arange(32, dtype=torch.float32).reshape(8, 2, 2)
        policy = TensorParallel(ranks=8, tp_size=4, axis=0)
        direct = policy.apply("weight", source, dtype=torch.float16, destination=None)
        destination = torch.empty_like(direct)

        result = policy.apply("weight", source, dtype=torch.float16, destination=destination)

        assert result.data_ptr() == destination.data_ptr()
        assert torch.equal(result, direct)

    def test_refuses_a_non_divisible_tensor_axis(self):
        policy = TensorParallel(ranks=8, tp_size=4, axis=0)

        with pytest.raises(ValueError, match="must divide"):
            policy.apply("weight", torch.ones(6, 2), dtype=None, destination=None)


class TestStagingPolicy:
    """Serial vs pooled staging, and the two properties that make pooling safe."""

    def test_serial_staging_visits_every_layer_in_order(self):
        seen: list[int] = []

        stage_layers([0, 1, 2], stage=seen.append, policy=StagingPolicy(workers=1))

        assert seen == [0, 1, 2]

    def test_a_single_worker_uses_no_pool_at_all(self):
        """`workers=1` must not merely be a pool of one: it runs on the calling thread.

        A family that must not overlap should not be able to overlap by scheduling accident.
        """
        threads: set[int] = set()

        stage_layers(
            [0, 1, 2],
            stage=lambda _: threads.add(threading.get_ident()),
            policy=StagingPolicy(workers=1),
        )

        assert threads == {threading.get_ident()}

    def test_pooled_staging_visits_every_layer(self):
        lock = threading.Lock()
        seen: list[int] = []

        def _stage(layer_id: int) -> None:
            with lock:
                seen.append(layer_id)

        stage_layers(list(range(16)), stage=_stage, policy=StagingPolicy(workers=4))

        assert sorted(seen) == list(range(16))

    def test_pooled_staging_actually_overlaps(self):
        """Otherwise the policy is a no-op that reads like an optimisation."""
        started = threading.Barrier(3, timeout=10)

        # Each worker waits for two others to arrive; a serial runner would deadlock here, so
        # reaching the assert at all is the evidence.
        stage_layers([0, 1, 2], stage=lambda _: started.wait(), policy=StagingPolicy(workers=3))

        assert started.parties == 3

    def test_a_failing_layer_propagates_rather_than_being_swallowed(self):
        """A half-written slab must not be handed back as if it were complete."""

        def _stage(layer_id: int) -> None:
            if layer_id == 2:
                raise RuntimeError("layer 2 is broken")

        with pytest.raises(RuntimeError, match="layer 2 is broken"):
            stage_layers([0, 1, 2, 3], stage=_stage, policy=StagingPolicy(workers=2))

    def test_torch_thread_count_is_restored_after_pinning(self):
        """Pinning is for the staging window only; leaking it would slow the whole process."""
        before = torch.get_num_threads()

        stage_layers([0, 1], stage=lambda _: None, policy=StagingPolicy(workers=2))

        assert torch.get_num_threads() == before

    def test_thread_pinning_touches_process_state_once_around_the_pool(self, monkeypatch):
        """Pinning must be set once around the pool, not saved and restored per worker.

        `torch.set_num_threads` is process-wide, so a per-worker save/restore races with
        itself: worker A saves 4 and sets 1, worker B then saves *1*, A restores 4, and B --
        finishing last -- restores 1, leaving the serving process single-threaded for
        everything after staging.

        The interleaving that does it cannot be forced from `stage`, because each worker reads
        the count before its callback runs; a test built on events inside `stage` passes on the
        broken code too. So this asserts the protocol instead: exactly one pin and one restore
        for the whole pool, whatever the worker count.
        """
        real_get = torch.get_num_threads
        sets: list[int] = []
        monkeypatch.setattr(torch, "set_num_threads", sets.append)

        stage_layers([0, 1, 2, 3], stage=lambda _: None, policy=StagingPolicy(workers=2))

        assert sets == [1, real_get()]

    def test_pinning_is_restored_even_when_a_layer_raises(self):
        """A staging failure must not leave the process pinned to one thread."""
        before = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                stage_layers(
                    [0, 1],
                    stage=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
                    policy=StagingPolicy(workers=2),
                )
            assert torch.get_num_threads() == 4
        finally:
            torch.set_num_threads(before)

    def test_zero_workers_is_refused(self):
        with pytest.raises(ValueError, match="at least one worker"):
            StagingPolicy(workers=0)

    def test_on_layer_done_fires_once_per_layer(self):
        done: list[int] = []

        stage_layers(
            [0, 1, 2],
            stage=lambda _: None,
            policy=StagingPolicy(workers=1),
            on_layer_done=done.append,
        )

        assert done == [0, 1, 2]


class TestStageAndRelease:
    """Load, write, drop — the shape that keeps peak memory at one layer per worker."""

    def test_each_layer_is_written_from_what_was_loaded_for_it(self):
        written: dict[int, str] = {}

        stage_and_release(
            [0, 1, 2],
            load=lambda layer_id: {"w": torch.full((1,), float(layer_id))},
            write=lambda layer_id, raw: written.__setitem__(layer_id, raw["w"].item()),
            policy=StagingPolicy(workers=1),
        )

        assert written == {0: 0.0, 1: 1.0, 2: 2.0}

    def test_a_layers_tensors_are_released_once_its_write_returns(self):
        """The release is by scope, so nothing may hold a layer's tensors past the call.

        Asserted through weak references to the tensors — the memory that actually matters —
        rather than by reading the code, because "we do not keep it" is exactly the kind of
        claim that quietly stops being true. A plain dict cannot be weakly referenced, which is
        itself why this watches the tensors and not the mapping.
        """
        import gc
        import weakref

        refs: list[weakref.ReferenceType] = []

        def _load(layer_id: int):
            tensor = torch.zeros(4)
            refs.append(weakref.ref(tensor))
            return {"w": tensor}

        stage_and_release(
            [0, 1, 2],
            load=_load,
            write=lambda *_: None,
            policy=StagingPolicy(workers=1),
        )
        gc.collect()

        assert len(refs) == 3
        # No layer survives staging: peak stays at one layer, not one per layer.
        assert [index for index, ref in enumerate(refs) if ref() is not None] == []
