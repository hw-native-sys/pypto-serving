# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin


def test_l3_dispatch_forwards_model_run_config_to_sync_and_async_calls():
    run_config = object()
    calls = []

    class Handle:
        @staticmethod
        def result():
            return None

    class Worker:
        def run(self, *args, **kwargs):
            calls.append(("run", args, kwargs))

        def submit(self, *args, **kwargs):
            calls.append(("submit", args, kwargs))
            return Handle()

    class Runner(L3DispatchMixin):
        def __init__(self):
            self._init_l3_dispatch(stacked=True)
            self._l3_worker = Worker()

        def _shared_l3_worker(self):
            return self._l3_worker

        def _l3_run_config(self, callable_spec):
            return run_config

    runner = Runner()
    callable_spec = SimpleNamespace(
        compiled=object(),
        name="kernel",
        aicpu_thread_num=4,
        block_dim=None,
        dispatch_args=(),
    )

    runner._run_l3(callable_spec)
    runner._submit_l3(callable_spec).wait()

    assert [kind for kind, _args, _kwargs in calls] == ["run", "submit"]
    assert all(kwargs == {"config": run_config} for _kind, _args, kwargs in calls)
