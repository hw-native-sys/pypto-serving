# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GLM TP16/EP16 executor using the kernel-owned resident forward session."""

from pathlib import Path

from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor
from pypto_serving.model.glm53.kernel_api import import_kernel
from pypto_serving.model.glm53.npu_runner import Glm53ModelRunner


class Glm53PyptoExecutor(PyptoExecutor):
    @property
    def max_prefill_batch_size(self):
        return 1

    def _compile_model(self, model):
        if model.extra.get("family") != "glm53":
            raise ValueError("Glm53PyptoExecutor requires a GLM-5.3-Flash W8A8 checkpoint")
        if len(self._device_ids) != 16 or len(set(self._device_ids)) != 16:
            raise ValueError("GLM-5.3-Flash requires 16 distinct NPU device ids")
        api = import_kernel("prefill_fwd", capacity=int(model.extra["capacity"]))
        runtime_dir = Path(self._pypto_build_dir or "build_output/glm53-serving")
        programs = api.compile_production(runtime_dir, list(self._device_ids), platform=self._platform)
        return api, programs, api.Checkpoint(model.extra["model_dir"])

    def _create_runner(self, model_id, compiled):
        api, programs, checkpoint = compiled
        return Glm53ModelRunner(api, programs, checkpoint, self._device_ids, self._platform)

    def lookup_embeddings(self, model, token_ids):
        session = self._runners[model.config.model_id]._session
        return session.weights.lookup_rows(token_ids)

    def release_finished_requests(self, request_ids):
        for runner in self._runners.values():
            runner.release_finished_requests(request_ids)
