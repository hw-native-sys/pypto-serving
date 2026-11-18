# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Restore immutable checkpoints into exclusive, stable request rings."""

import hashlib
import json
from math import prod
from pathlib import Path

import torch

from pypto_serving.serving.memory.protocol import CachePublicationProof
from .prefix_protocol import PrefixResumeDescriptor
from .cache import TENSOR_GROUPS
from .profiling import operation


def cache_namespace(model_dir, *, context, chunk):
    """One manager owns one model/profile; reload naturally creates a new namespace."""
    model_dir = Path(model_dir).resolve()
    digest = hashlib.sha256(
        json.dumps(
            dict(
                model=str(model_dir),
                context=context,
                chunk=chunk,
                topology="B1-TP8-CP8-EP1",
                abi=(81, 100, 153),
                layout="rolling-v2",
            ),
            sort_keys=True,
        ).encode()
    )
    for name in ("config.json", "model.safetensors.index.json"):
        path = model_dir / name
        if path.exists():
            digest.update(path.read_bytes())
    lib = Path(__file__).resolve().parents[3] / "pypto-lib/models/deepseek_v4_flash_lowlat"
    for path in sorted(lib.rglob("*.py")):
        digest.update(str(path.relative_to(lib)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class PrefixPageCopies:
    """Bounded registered HOST staging, reused only with no accepted device work."""

    def __init__(self):
        self.host = torch.empty(128 * 512 * 2, dtype=torch.uint8).share_memory_()
        self.bytes_copied = 0

    @operation("lowlat.prefix.copy", category="lowlat")
    def copy(self, cache, *, source, destination, ranges):
        if cache.arena is None:
            raise RuntimeError("Prefix copy requires allocated Worker-owned cache")
        plans = []
        with cache.programs.context():
            from models.deepseek_v4_flash_lowlat.prefill_cache import CACHE_LAYOUTS

            layouts = dict(CACHE_LAYOUTS)
        for interval in ranges:
            group = interval.group
            spec = cache.specs[group]
            if spec.sliding_window is None:
                if tuple(source[group][: interval.end]) != tuple(destination[group][: interval.end]):
                    raise ValueError("Full-history checkpoint pages must remain shared and immutable")
                continue
            old, new = source[group], destination[group]
            if not old or not new:
                raise ValueError("Prefix copy needs source and destination rings")
            pairs = [(old[i % len(old)], new[i % len(new)]) for i in range(interval.start, interval.end)]
            if any(a < 0 or b < 0 or a >= spec.num_blocks or b >= spec.num_blocks for a, b in pairs) or set(
                a for a, _ in pairs
            ) & set(new):
                raise ValueError("Prefix copy must use disjoint live source and destination pages")
            for tensor, owner in cache.owners.items():
                if TENSOR_GROUPS[tensor] != group:
                    continue
                layers, rows, _, tail, dtype = layouts[tensor]
                page_bytes = rows * prod(tail) * torch.empty((), dtype=dtype).element_size()
                if page_bytes > self.host.numel():
                    raise ValueError("Prefix page exceeds registered copy staging")
                for rank, shard in enumerate(owner.shards):
                    for layer in range(layers):
                        for a, b in pairs:
                            base = shard.data_ptr - shard.byte_offset
                            layer_offset = shard.byte_offset + layer * spec.num_blocks * page_bytes
                            plans.append(
                                (
                                    rank,
                                    base,
                                    layer_offset + a * page_bytes,
                                    layer_offset + b * page_bytes,
                                    page_bytes,
                                    tensor,
                                    b,
                                )
                            )
        # Worker transfers retain the allocation owner; offsets describe views.
        for rank, base, source_offset, destination_offset, nbytes, tensor, page in plans:
            cache.arena.worker.copy_from(
                self.host.data_ptr(), base, nbytes, src_offset=source_offset, worker_id=rank
            )
            cache.arena.worker.copy_to(
                base, self.host.data_ptr(), nbytes, dst_offset=destination_offset, worker_id=rank
            )
            cache.initialized[tensor][rank].add(page)
            self.bytes_copied += nbytes


def validate_resume(descriptor, *, namespace, context, cache):
    if not isinstance(descriptor, PrefixResumeDescriptor):
        raise TypeError("Lowlat prefix resume requires a typed descriptor")
    descriptor.__post_init__()
    if (
        descriptor.namespace != namespace
        or descriptor.request_id != context.request_id
        or descriptor.execution_generation != context.execution_generation
        or descriptor.hit_tokens + descriptor.replay_tokens != context.prompt_tokens
        or set(descriptor.tables) != set(cache.specs)
    ):
        raise ValueError("Lowlat prefix resume differs from admitted model/request")
    expected = cache.checkpoint_ranges(descriptor.hit_tokens)
    if descriptor.valid_ranges != expected:
        raise ValueError("Lowlat prefix resume does not prove the full joint state window")
    ids = {(p.group, p.page_id): p for p in descriptor.pages}
    required = set()
    for interval in expected:
        table = descriptor.tables[interval.group]
        if not table:
            raise ValueError("Lowlat prefix resume has an empty group")
        for logical in range(interval.start, interval.end):
            slot = logical if cache.specs[interval.group].sliding_window is None else logical % len(table)
            page = table[slot]
            identity = ids.get((interval.group, page))
            if identity is None or identity.partition != 0:
                raise ValueError("Lowlat prefix resume is missing a required page identity")
            required.add((interval.group, page))
    if required != set(ids):
        raise ValueError("Lowlat prefix resume includes unproven pages")
    cache.adopt_page_identities(descriptor.pages, retained=True)
    return descriptor.hit_tokens


def checkpoint_proof(plan):
    return CachePublicationProof(
        plan.request_id,
        plan.generation,
        plan.mapping_version,
        plan.boundary,
        plan.boundary,
        plan.complete_logical_pages,
        plan.destination_pages,
    )
