# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load real TP4/EP4 native drafter and Markov checkpoint weights on CPU."""

from contextlib import ExitStack

import torch
from pypto.runtime import ReadOnlyHostTensor
from pypto_serving.model.common.weights.nz import pack_nz
from pypto_serving.model.deepseek_dspark.weight_loader import DSparkWeightStore
from pypto_serving.model.deepseek_dspark.weight_spec import dspark_drafter_required_weight_names

from .bindings import DTYPES, parameter_rows


class NativeDraftWeights:
    def __init__(self, model, metadata):
        self.store = DSparkWeightStore(model_dir=model.extra["model_dir"],
                                      weight_map=model.extra["weight_map"])
        self.rows = parameter_rows(metadata)
        self.hosts = {}

    def prepare(self):
        if self.hosts:
            raise RuntimeError("native draft weights were already prepared")
        required = dspark_drafter_required_weight_names(256)
        self.store.require(required)
        # Reject silent casts of quantized values before the existing packer copies them.
        checked = 0
        with ExitStack() as stack:
            readers = {}
            for name in required:
                quantized = (name.startswith("mtp.") and name.endswith(".weight")
                             and (".ffn.experts." in name or ".ffn.shared_experts." in name
                                  or name.endswith(".attn.wq_b.weight")))
                if not quantized:
                    continue
                path = self.store.path_for(name)
                if path not in readers:
                    readers[path] = stack.enter_context(self.store._safe_open_fn(path, "cpu"))
                source = readers[path].get_slice(name)
                if source.get_dtype() != "I8":
                    raise ValueError("native draft values must be original INT8: "+name)
                checked += 1
        packed = self.store.load_drafter_weights(ranks=4, n_routed_experts=256, tp_size=4)
        prepared = {}
        for name, value in packed.tensors.items():
            matches = [prefix+name for prefix in ("d_", "m_") if prefix+name in self.rows]
            if len(matches) != 1:
                raise ValueError("ambiguous/missing native D/M weight: "+name)
            prepared[matches[0]] = value
        embedding = self.store.load_tensor("embed.weight")
        head = self.store.load_tensor("head.weight")
        if any(tuple(value.shape) != (129280, 4096) or value.dtype != torch.bfloat16
               for value in (embedding, head)):
            raise ValueError("native D/M requires exact original BF16 vocabulary tables")
        # All four ranks read the same inherited CPU embedding backing.
        prepared["d_embedding_weight"] = embedding.clone().share_memory_().unsqueeze(0).expand(4, -1, -1)
        prepared["m_lm_head_weight"] = torch.stack([pack_nz(head[r*32320:(r+1)*32320]) for r in range(4)])
        for name, value in prepared.items():
            row = self.rows[name]
            if (tuple(value.shape) != tuple(row["shape"]) or value.dtype != DTYPES[row["dtype"]]
                    or row["direction"] != "In" or value.shape[0] != 4):
                raise ValueError("real D/M weight does not match the compiled ABI: "+name)
            if name != "d_embedding_weight":
                value.share_memory_()
        self.hosts = prepared
        return dict(quantized_source_headers_checked=checked, weight_banks=len(prepared),
                    bytes_per_drafter_rank=self.bytes_for_rank(0), original_int8_preserved=True,
                    drafter_tp=4, drafter_ep=4, hc_rows_per_layer=24,
                    embedding_shared_host_backing=True, markov_head_layout="NZ")

    @property
    def inherited(self):
        return tuple(ReadOnlyHostTensor(value[rank]) for value in self.hosts.values() for rank in range(4))

    def bytes_for_rank(self, rank):
        if type(rank) is not int or not 0 <= rank < 8:
            raise ValueError("invalid target rank")
        return sum(v[rank].numel()*v.element_size() for v in self.hosts.values()) if rank < 4 else 0

    def upload(self, arena):
        if not self.hosts:
            raise RuntimeError("prepare real native weights before constructing the Worker")
        return {name: arena.upload(value) for name, value in self.hosts.items()}
