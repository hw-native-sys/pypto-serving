# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-only execution policies selected at the model integration boundary."""


class ExecutionPolicy:
    requires_token_budget = False

    def __init__(self, config):
        self.config = config

    def resolve_async_scheduling(self):
        return self.config.async_scheduling if self.config.async_scheduling is not None else True

    def validate_min_tokens(self, value):
        pass

    def validate_generate_config(self, config, *, constraint_spec=None):
        pass

    def validate_request_budget(self, prompt_tokens, output_tokens):
        pass

    def scheduler_options(self):
        return {}

    def confirm_step(self, output, result, cache):
        if result.execution_receipts or result.retired_reservation_ids or result.cache_publications:
            raise ValueError("Executor returned undeclared execution receipts")

    def request_metadata(self, request):
        return b""

    def prefill_metadata(self, scheduled, *, is_new, cache):
        return b""


def execution_policy(config):
    if config.executor_cls == "PyptoDeepSeekV4LowlatExecutor":
        from .deepseek_lowlat.execution_policy import LowlatExecutionPolicy

        return LowlatExecutionPolicy(config)
    return ExecutionPolicy(config)
