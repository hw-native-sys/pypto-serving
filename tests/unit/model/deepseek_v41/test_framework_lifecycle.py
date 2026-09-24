# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""8K/128 framework regression with a recording adapter, not M0 numerical evidence.

No model kernels, physical cache allocator, HTTP server or NPU are executed.
The adapter records ownership/reset callbacks and emits synthetic next-token
logits; full-history test pages do not claim the model's rolling SWA cache ABI.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from pypto_serving.config.types import (
    DecodeBatch, KVCacheGroupSpec, KVCacheSpec, RuntimeConfig, SamplingParams,
)
from pypto_serving.model.common.executor.sampler import Sampler
from pypto_serving.model.deepseek_v41.composite import CompositeBindings, LayerState
from pypto_serving.model.deepseek_v41.config import V41TextConfig
from pypto_serving.model.deepseek_v41.execution_plan import RankPlacement, plan_layers
from pypto_serving.model.deepseek_v41.npu_runner import V41ModelRunner
from pypto_serving.serving.utils.prefill import pack_prefill_batch


def test_public_runner_chunked_8k_128_decode_release_and_reuse():
    raw = json.loads((Path(__file__).resolve().parents[4] /
                      "tests/fixtures/deepseek_v41/config.json").read_text(encoding="utf-8"))
    raw["text_config"].update(hidden_size=4, vocab_size=16, max_position_embeddings=8320)
    config = V41TextConfig.from_dict(raw)
    layers = plan_layers(raw)
    assert len(layers) == 40
    assert {layer.mode for layer in layers} == {
        "swa", "c2a_full", "c2a_reuse", "c1a_full", "c1a_reindex", "c1a_reuse",
    }
    groups = (
        KVCacheGroupSpec("window", tuple(range(40)), KVCacheSpec(128, 16),
                         65, num_blocks=65, num_partitions=2),
        KVCacheGroupSpec("compressed", (2, 8, 14, 20), KVCacheSpec(256, 16, 2),
                         33, num_blocks=33, num_partitions=2),
    )
    runtime = RuntimeConfig(max_batch_size=2, max_seq_len=8320, max_num_batched_tokens=2048,
                            max_prefill_tokens_per_request=1024, kv_cache_groups=groups)
    table = torch.arange(64, dtype=torch.float32).reshape(16, 4).bfloat16()
    resources = {"slots": {}, "pages": {}}
    observed_positions, observed_slots, steps, layer_calls, resets = {}, {}, [], [], []
    completion = {"pending": False, "closed": False}

    def embeddings(ids):
        return table.index_select(0, ids.reshape(-1))

    def initialize(values, step, allocated):
        assert allocated is resources
        assert not completion["pending"]
        assert torch.equal(values, embeddings(torch.tensor(step.token_ids)))
        steps.append(step)
        for request in step.requests:
            key = request.request_id
            history = observed_positions.setdefault(key, [])
            assert request.start == len(history)
            history.extend(range(request.start, request.end))
            slot = (request.partition, request.state_slot)
            assert observed_slots.setdefault(key, slot) == slot
            assert resources["slots"].setdefault(slot, key) == key
            for name, pages in request.pages.items():
                for page in pages:
                    assert resources["pages"].setdefault((request.partition, name, page), key) == key
        completion["pending"] = True
        return LayerState(object(), object())

    def prepare(plans, layer, allocated):
        assert not completion["pending"]
        assert [(plan.placement.tp_rank, plan.placement.dp_rank) for plan in plans] == [
            (rank % 4, rank // 4) for rank in range(8)
        ]
        return layer.layer_id

    def call(layer, state, step, allocated, weights):
        assert not completion["pending"]
        assert weights == layer.layer_id
        layer_calls.append((step.epoch, step.phase, layer))
        completion["pending"] = True
        return state

    def output(state, step, allocated):
        assert not completion["pending"]
        logits = torch.full((len(step.requests), 16), -1.0)
        for row, request in enumerate(step.requests):
            logits[row, (request.token_ids[-1] + 1) % 16] = 1.0
        completion["pending"] = True
        return logits

    def reset(allocated, key, owner):
        assert not completion["pending"]
        slot = (owner.partition, owner.slot)
        assert resources["slots"].pop(slot) == key
        pages = {(owner.partition, name, page) for name, ids in owner.pages.items() for page in ids}
        assert pages == {address for address, value in resources["pages"].items() if value == key}
        for address in pages:
            assert resources["pages"].pop(address) == key
        resets.append((key, slot, pages))
        completion["pending"] = True

    def wait(allocated):
        completion["pending"] = False

    def close(allocated):
        assert not completion["pending"] and not any(resources.values())
        completion["closed"] = True

    bindings = CompositeBindings(
        revision="recording-lifecycle-test-only", cache_groups=groups,
        entries={(phase, layer.mode): call for phase in ("prefill", "decode") for layer in layers},
        initialize=initialize, output=output, allocate=lambda *args: (resources, 65),
        prepare_weights=prepare, reset_request=reset, wait=wait, close=close,
    )
    plan = SimpleNamespace(placement=RankPlacement(0), layers=layers, weights=SimpleNamespace(config=config))
    plan.for_rank = lambda rank: SimpleNamespace(placement=RankPlacement(rank))
    runner = V41ModelRunner(plan, bindings, device_ids=(7, 2, 5, 0, 6, 1, 4, 3), runtime=runtime)
    model = SimpleNamespace(config=config)
    partitions, lengths = {"A": 1, "B": 0, "C": 1}, {"A": 0, "B": 0}
    prompts = {"A": [(i + 3) % 16 for i in range(8192)],
               "B": [(i + 7) % 16 for i in range(8192)]}

    def pages(end):
        return {group.name: list(range((end + group.spec.token_capacity - 1) // group.spec.token_capacity))
                for group in groups}

    def prefill(order, size):
        ends = [lengths[key] + size for key in order]
        batch = pack_prefill_batch(
            request_ids=order, token_chunks=[prompts[key][lengths[key]:end] for key, end in zip(order, ends)],
            seq_lens=ends, chunk_starts=[lengths[key] for key in order],
            prompt_lens=[len(prompts[key]) for key in order], device="cpu", embedding_lookup=embeddings,
            cache_partitions=[partitions[key] for key in order],
            block_ids_by_group=[pages(end) for end in ends],
        )
        result = runner.run_prefill(model, batch)
        lengths.update(zip(order, ends))
        return result

    # A pauses while B advances, then B pauses while A catches up. Both retain
    # their state and partition-local page IDs despite changing packed row order.
    orders = [("A", "B"), ("B",), ("B", "A"), ("A",)]
    orders += [("B", "A") if i % 2 else ("A", "B") for i in range(5)]
    for order in orders:
        omitted = {key: owner.length for key, owner in (runner.ledger.owners.items() if runner.ledger else [])
                   if key not in order}
        result = prefill(order, 1024)
        assert all(runner.ledger.owners[key].length == length for key, length in omitted.items())
    assert lengths == {"A": 8192, "B": 8192}
    assert steps[-1].terminal_prefill == (True, True)
    assert not any(terminal for step in steps[:-1] for terminal in step.terminal_prefill)

    sampler, params = Sampler(), SamplingParams(temperature=0.0, top_p=1.0)
    next_token = {key: sampler.sample(result.logits[row], params, key) for row, key in enumerate(order)}
    generated = {key: [] for key in prompts}
    for turn in range(128):
        order = ("B", "A") if turn % 2 else ("A", "B")
        ids = torch.tensor([[next_token[key]] for key in order])
        ends = [lengths[key] + 1 for key in order]
        batch = DecodeBatch(
            request_ids=list(order), token_ids=ids, hidden_states=embeddings(ids),
            seq_lens=torch.tensor(ends), cache_partitions=[partitions[key] for key in order],
            block_ids_by_group=[pages(end) for end in ends],
        )
        result = runner.run_decode(model, batch)
        for row, key in enumerate(order):
            generated[key].append(next_token[key])
            next_token[key] = sampler.sample(result.logits[row], params, key)
        lengths.update(zip(order, ends))
    for key in prompts:
        assert generated[key] == [(prompts[key][-1] + i + 1) % 16 for i in range(128)]
        assert observed_positions[key] == list(range(8320))
        assert runner.ledger.owners[key].length == 8320
    assert observed_slots == {"A": (1, 0), "B": (0, 0)}
    assert len(layer_calls) == len(steps) * 40
    for index, step in enumerate(steps):
        assert layer_calls[index * 40:(index + 1) * 40] == [
            (step.epoch, step.phase, layer) for layer in layers
        ]
    assert [step.epoch for step in steps] == list(range(1, len(steps) + 1))

    runner.release_finished_requests(["A", "A"])
    assert [event[0] for event in resets] == ["A"]
    assert "A" not in runner.ledger.owners and runner.ledger.owners["B"].length == 8320
    assert resets[0][2] == {(1, name, page) for name, ids in pages(8320).items() for page in ids}
    assert set(resources["pages"].values()) == {"B"}
    prompts["C"], lengths["C"] = [4, 5, 6], 0
    prefill(("C",), 3)
    assert observed_slots["C"] == observed_slots["A"]
    assert observed_positions["C"] == [0, 1, 2]
    assert resources["pages"][(1, "window", 0)] == "C"
    assert resources["pages"][(0, "window", 0)] == "B"
    runner.release_finished_requests(["C", "B"])
    assert [event[0] for event in resets] == ["A", "C", "B"]
    assert not runner.ledger.owners and not any(resources.values())
    runner.close()
    assert completion["closed"]
