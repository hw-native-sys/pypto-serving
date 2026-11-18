# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read-only prefix diagnostics for the skill's dedicated acceptance server."""

from dataclasses import asdict
from functools import wraps
import hashlib
import json
from math import prod
import os
from pathlib import Path

import torch


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _read(worker, shard, offset, nbytes, scratch, rank):
    if scratch.dtype != torch.uint8 or not scratch.is_contiguous() or not scratch.is_shared():
        raise ValueError("Audit readback needs contiguous pre-fork shared bytes")
    if offset < 0 or nbytes < 0 or offset + nbytes > prod(shard.shape) * shard.dtype.itemsize:
        raise ValueError("Audit read exceeds its borrowed view")
    base = shard.data_ptr - shard.byte_offset
    for start in range(0, nbytes, scratch.numel()):
        size = min(scratch.numel(), nbytes - start)
        worker.copy_from(scratch.data_ptr(), base, size,
                         src_offset=shard.byte_offset + offset + start, worker_id=rank)
        yield scratch[:size].numpy().tobytes()


def cache_hashes(cache, tables, ranges, scratch):
    """Hash complete logical pages in canonical tensor/rank/layer/page order."""
    from pypto_serving.model.deepseek_lowlat.cache import TENSOR_GROUPS

    intervals = {interval.group: interval for interval in ranges}
    if intervals.keys() != cache.specs.keys() or tables.keys() != cache.specs.keys():
        raise ValueError("Audit requires every cache group")
    plans = []
    for name, owner in sorted(cache.owners.items()):
        group = TENSOR_GROUPS[name]
        spec = cache.specs[group]
        interval, table = intervals[group], tables[group]
        if (interval.start < 0 or interval.end < interval.start or not table
                or owner.shape[1] % spec.num_blocks or owner.shape[0] != 8 or len(owner.shards) != 8):
            raise ValueError("Invalid audit cache geometry")
        pages = [table[logical % len(table)] for logical in range(interval.start, interval.end)]
        if len(set(pages)) != len(pages) or any(not 0 <= page < spec.num_blocks for page in pages):
            raise ValueError("Audit live logical pages must have distinct valid storage")
        page_bytes = prod(owner.shape[2:]) * owner.dtype.itemsize
        plans.append((name, owner, spec.num_blocks, pages, page_bytes))
    result = {}
    for name, owner, blocks, pages, page_bytes in plans:
        ranks = []
        for rank, shard in enumerate(owner.shards):
            digest = hashlib.sha256()
            for layer in range(owner.shape[1] // blocks):
                for page in pages:
                    for part in _read(cache.arena.worker, shard, (layer * blocks + page) * page_bytes,
                                      page_bytes, scratch, rank):
                        digest.update(part)
            ranks.append(digest.hexdigest())
        result[name] = ranks
    return result


def owner_hashes(worker, owner, scratch):
    return [hashlib.sha256(b"".join(_read(worker, shard, 0,
            prod(shard.shape) * shard.dtype.itemsize, scratch, rank))).hexdigest()
            for rank, shard in enumerate(owner.shards)]


def _save(directory, record):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (record["request_id"] + ".json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def install(directory):
    """Install diagnostics only in the explicitly selected acceptance entry."""
    from pypto_serving.model.deepseek_lowlat.npu_runner import LowlatModelRunner
    from pypto_serving.model.deepseek_lowlat.native_runner import NativeModelRunner
    from pypto_serving.model.deepseek_lowlat.native_pipeline_runner import NativePipelineRunner
    from pypto_serving.model.deepseek_lowlat.prefix import PrefixPageCopies

    if getattr(LowlatModelRunner, "_prefix_audit_installed", False):
        return
    LowlatModelRunner._prefix_audit_installed = True
    records = {}

    def record_for(runner):
        state = runner.state
        if not runner.prefix_namespace or state is None or state.request_id.startswith("__"):
            return None
        if state.request_id not in records:
            context = runner.request_contexts[state.request_id]
            records[state.request_id] = dict(request_id=state.request_id,
                prompt_tokens=context.prompt_tokens, output_tokens=context.max_output_tokens,
                generation=state.generation, copies=[], token_ids=[], complete=False,
                scope="Complete logical cache pages, terminal128-tap seed and all native seed ranks; diagnostic only")
        return records[state.request_id]

    original_prefill = LowlatModelRunner.run_prefill

    @wraps(original_prefill)
    def prefill(runner, model, batch):
        runner._request(batch.request_ids, batch.prompt_lens[0])
        runner.cache._prefix_audit_record = record_for(runner)
        return original_prefill(runner, model, batch)

    LowlatModelRunner.run_prefill = prefill
    original_copy = PrefixPageCopies.copy

    @wraps(original_copy)
    def copy(copier, cache, *, source, destination, ranges):
        result = original_copy(copier, cache, source=source, destination=destination, ranges=ranges)
        record = getattr(cache, "_prefix_audit_record", None)
        if record is not None:
            before = cache_hashes(cache, source, ranges, copier.host)
            after = cache_hashes(cache, destination, ranges, copier.host)
            if before != after:
                raise RuntimeError("Prefix audit: copied logical state differs")
            checkpoint = all(tuple(source[name]) == tuple(ids) for name, ids in cache._page_ids.items())
            record["copies"].append(dict(kind="checkpoint" if checkpoint else "restore",
                ranges=[asdict(r) for r in ranges], hashes=before))
            _save(directory, record)
        return result

    PrefixPageCopies.copy = copy
    original_finalize = NativeModelRunner.finalize_prefill

    @wraps(original_finalize)
    def finalize(runner, *args, **kwargs):
        result = original_finalize(runner, *args, **kwargs)
        record = record_for(runner)
        if record is not None:
            if runner.native.state is None or getattr(runner, "_accepted", ()):
                raise RuntimeError("Prefix audit needs an idle native seed")
            scratch = runner.prefix_copies.host
            record["prefill_cache"] = cache_hashes(runner.cache, runner.cache._page_ids,
                runner.cache.checkpoint_ranges(runner.state.prompt_len), scratch)
            history = runner.native.history.host
            seed = tensor_hash(history)
            history_hashes = owner_hashes(runner.worker, runner.native.common["s_history_ring"], scratch)
            zero = hashlib.sha256(bytes(history.numel() * history.element_size())).hexdigest()
            if history_hashes != [seed, *([zero] * 7)]:
                raise RuntimeError("Prefix audit: device history seed differs from real captured taps")
            record["history_seed"] = seed
            record["device_history"] = history_hashes
            state_owner = runner.native.common["s_request_state"]
            expected = runner.native.inputs.state
            expected_hashes = [tensor_hash(row) for row in expected]
            if owner_hashes(runner.worker, state_owner, scratch) != expected_hashes:
                raise RuntimeError("Prefix audit: one or more native seed ranks differ")
            record["native_seed"] = expected.tolist()
            record["token_ids"] = [int(runner.state.last_sampled_token)]
            _save(directory, record)
        return result

    NativeModelRunner.finalize_prefill = finalize
    original_retire = NativePipelineRunner.reclaim_prepared_decode

    @wraps(original_retire)
    def retire(runner, *args, **kwargs):
        result = original_retire(runner, *args, **kwargs)
        record = record_for(runner)
        if record is not None:
            record["token_ids"].extend(result.accepted_token_ids[0])
            record["complete"] = len(record["token_ids"]) == record["output_tokens"]
            _save(directory, record)
        return result

    NativePipelineRunner.reclaim_prepared_decode = retire


def validate(directory):
    from pypto_serving.model.deepseek_lowlat.cache import TENSOR_GROUPS

    records = [json.loads(path.read_text()) for path in Path(directory).glob("*.json")
               if path.name != "validation.json"]
    cold = [r for r in records if [c["kind"] for c in r["copies"]] == ["checkpoint"]]
    hot = [r for r in records if [c["kind"] for c in r["copies"]] == ["restore"]]
    if len(cold) != 1 or len(hot) != 2 or len(records) != 3:
        raise ValueError("Expected one complete cold checkpoint and two hot restores")
    baseline = cold[0]
    for record in records:
        for hashes in (record["prefill_cache"], record["copies"][0]["hashes"]):
            if hashes.keys() != TENSOR_GROUPS.keys() or any(len(ranks) != 8 for ranks in hashes.values()):
                raise ValueError("Missing cache family or rank")
        if (not record["complete"] or len(record["token_ids"]) != record["output_tokens"]
                or any(record[key] != baseline[key] for key in (
                    "prompt_tokens", "output_tokens", "token_ids", "prefill_cache", "history_seed", "device_history"))
                or record["copies"][0]["hashes"] != baseline["copies"][0]["hashes"]
                or record["copies"][0]["ranges"] != baseline["copies"][0]["ranges"]):
            raise ValueError("Cold/hot IDs, live pages, seed, or retained checkpoint differ")
        rows = record["native_seed"]
        if len(rows) != 8 or any(row != rows[0] for row in rows):
            raise ValueError("Native seed ranks differ")
        if rows[0][1:] != [record["prompt_tokens"] - 1, record["output_tokens"] - 1, record["token_ids"][0]]:
            raise ValueError("Native seed does not describe the actual request")
    result = dict(valid=True, requests=3, cold=1, hot=2,
                  checks=["exact_output_ids", "complete_logical_cache_pages_all_ranks",
                          "checkpoint_after_prior_decode", "terminal_history_seed", "native_seed_all_ranks"],
                  limitation="Does not qualify partial pages, numerical golden, cancellation, eviction or long context")
    (Path(directory) / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if os.environ.get("LOWLAT_PREFIX_AUDIT_DIR"):
    install(os.environ["LOWLAT_PREFIX_AUDIT_DIR"])

if __name__ == "__main__":
    from pypto_serving.cli.main import main

    raise SystemExit(main())
