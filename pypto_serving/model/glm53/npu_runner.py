# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Single-request GLM serving with resident weights and full-prefix forwards."""

from __future__ import annotations

import math

from pypto_serving.config.types import DecodeResult, PrefillResult
from pypto_serving.model.common.runner.model_runner import ModelRunner


class Glm53ModelRunner(ModelRunner):
    """Keep request token histories and execute one model step per callback."""

    def __init__(self, api, programs, checkpoint, device_ids, platform):
        super().__init__()
        self._api, self._programs, self._checkpoint = api, programs, checkpoint
        self._device_ids, self._platform = list(device_ids), platform
        self._session = None
        self._histories: dict[str, list[int]] = {}

    def init_kv_cache(self, model_id, config, runtime, *, num_pages=None):
        """Return scheduler admission capacity; each forward rebuilds device caches."""
        if runtime.max_batch_size != 1:
            raise ValueError("GLM full-prefix serving requires max_num_seqs=1")
        self._max_seq_len = runtime.max_seq_len
        return math.ceil(runtime.max_seq_len / runtime.page_size)

    def preflight(self, record):
        from pypto.ir import DistributedConfig
        from pypto.runtime import RunConfig

        options = {name: getattr(record.runtime, name) for name in ("ring_heap", "ring_task_window", "ring_dep_pool")}
        if options["ring_heap"] is None:
            options["ring_heap"] = 2 * 1024**3
        config = RunConfig(
            platform=self._platform,
            distributed_config=DistributedConfig(device_ids=self._device_ids, num_sub_workers=0),
            **options,
        )
        self._session = self._api.ResidentPrefillSession(
            self._checkpoint, self._programs, self._device_ids, platform=self._platform, run_config=config,
        )
        try:
            self._session.__enter__()
        except BaseException:
            self._session = None
            raise

    def _alloc_kv_cache_tensor(self, shape, dtype):
        return self._session.runtime.alloc_tensor(shape, dtype, worker_id=0)

    def _free_kv_cache_tensor(self, tensor):
        self._session.runtime.free_tensor(tensor)

    def _forward(self, tokens):
        if not tokens or len(tokens) > self._max_seq_len:
            raise ValueError("GLM request token history exceeds max_model_len or is empty")
        # Inactive slots carry one valid token and never enter sampling or output.
        sequences = [tokens] + [[0] for _ in range(self._api.BATCH - 1)]
        return self._session.forward(sequences)[:1]

    def run_prefill(self, model, batch):
        if len(batch.request_ids) != 1 or any(batch.chunk_starts):
            raise ValueError("GLM full-prefix serving requires one complete prompt per prefill")
        tokens = batch.token_ids.reshape(-1).tolist()
        if batch.chunk_lens != [len(tokens)] or batch.seq_lens != [len(tokens)]:
            raise ValueError("GLM full-prefix serving does not accept chunked or cached-prefix prefill")
        logits = self._forward(tokens)
        self._histories[batch.request_ids[0]] = tokens
        return PrefillResult(last_hidden=None, logits=logits)

    def run_decode(self, model, batch):
        if len(batch.request_ids) != 1 or batch.token_ids.numel() != 1:
            raise ValueError("GLM full-prefix serving requires one active decode request")
        request_id = batch.request_ids[0]
        if request_id not in self._histories:
            raise ValueError("GLM decode requires a completed prefill for this request")
        tokens = self._histories[request_id] + [int(batch.token_ids.item())]
        logits = self._forward(tokens)
        self._histories[request_id] = tokens
        return DecodeResult(hidden_states=None, logits=logits)

    def release_finished_requests(self, request_ids):
        for request_id in request_ids:
            self._histories.pop(request_id, None)

    def close(self):
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None
        self._histories.clear()
