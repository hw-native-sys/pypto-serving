# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test-only device snapshots for the V4.1 HTTP prefix replay guard."""

import json
import math
import os
from pathlib import Path

import torch


def install(report_dir):
    """Instrument the server process, without changing its model or scheduling."""
    from pypto_serving.model.deepseek_v41.npu_runner import V41ModelRunner

    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    counter = 0

    def snapshot(runner, requests, *, padding=False):
        cache = runner.resources.cache
        result = {}
        for request in requests:
            hit = request.shared_prefix_tokens
            if not hit or not request.pages:
                continue
            for group in (cache.plan.family_groups.c2a, cache.plan.family_groups.c1a):
                pages = list(request.pages[group][:hit // 128])
                if padding:
                    pages.append(cache.plan.group_blocks[group])
                families = cache.plan.group_sources[group]
                for field, family in (("compressed_cache_pool", "compressed"),
                                      ("index_cache_pool", "index"),
                                      ("index_cache_scale_pool", "index")):
                    for source in families[family]:
                        device = cache.view(field, source)
                        page_bytes = math.prod(device.shape[2:]) * torch.empty((), dtype=device.dtype).element_size()
                        rank = request.partition
                        shard = device.shards[rank]
                        for page in pages:
                            key = field, source, rank, page
                            if key in result:
                                continue
                            host = torch.empty(page_bytes, dtype=torch.uint8)
                            runner.resources.worker.copy_from(
                                host.data_ptr(), shard.data_ptr - shard.byte_offset, page_bytes,
                                src_offset=shard.byte_offset + page * page_bytes,
                                worker_id=device.worker_ids[rank],
                            )
                            result[key] = host
        return result

    def report(kind, requests, before, after):
        assert before.keys() == after.keys(), "prefix snapshot addresses changed"
        changed = [key for key in before if not torch.equal(before[key], after[key])]
        assert not changed, f"{kind} modified shared or padding pages: {changed}"
        record = {"kind": kind, "page_snapshots": len(before), "unchanged": True,
                  "fields": sorted({key[0] for key in before}),
                  "requests": [{"id": r.request_id if hasattr(r, "request_id") else "",
                                "partition": r.partition, "start": getattr(r, "start", None),
                                "end": getattr(r, "end", None), "hit": r.shared_prefix_tokens}
                               for r in requests]}
        with (destination / f"pages.{os.getpid()}.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")

    original_transaction = V41ModelRunner._run_transaction

    def transaction(runner, step, embeddings, **kwargs):
        nonlocal counter
        warm = [r for r in step.requests if r.shared_prefix_tokens]
        before = snapshot(runner, warm, padding=step.phase == "prefill")
        result = original_transaction(runner, step, embeddings, **kwargs)
        if warm:
            after = snapshot(runner, warm, padding=step.phase == "prefill")
            report(step.phase, warm, before, after)
        if step.phase == "prefill" and any(step.terminal_prefill):
            counter += 1
            torch.save({"requests": [{"id": r.request_id, "start": r.start, "end": r.end,
                                      "hit": r.shared_prefix_tokens} for r in step.requests],
                        "logits": result.logits.detach().clone()},
                       destination / f"logits.{os.getpid()}.{counter}.pt")
        return result

    original_reset = V41ModelRunner._reset_request

    def reset(runner, key, owner):
        before = snapshot(runner, [owner])
        result = original_reset(runner, key, owner)
        if before:
            report("reset", [owner], before, snapshot(runner, [owner]))
        return result

    V41ModelRunner._run_transaction = transaction
    V41ModelRunner._reset_request = reset
