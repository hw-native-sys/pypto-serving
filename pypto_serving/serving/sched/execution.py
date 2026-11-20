# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Model-independent scheduler extension points; page ownership stays in Scheduler."""


class RequestPolicyState:
    """Host-only model state; never serialized implicitly across IPC."""


class StepPolicyState:
    """Host-only plan retained until the scheduled step settles."""


class SchedulingPolicy:
    uses_model_execution_plan = False
    explicit_cache_publication = False
    serial_prefill = False

    def validate_config(self, config):
        pass

    def admit(self, scheduler, request):
        pass

    def reserve_execution_plan(self, scheduler, request):
        return None

    def adopt_prefix(self, scheduler, request):
        pass

    def prepare_steps(self, scheduler, output):
        pass

    def limit_prefill(self, request, limit):
        return limit

    def allocation_extent(self, config, request, extent):
        return extent

    def reset_waiting(self, request):
        pass

    def confirmed_prefill(self, scheduler, request):
        pass
