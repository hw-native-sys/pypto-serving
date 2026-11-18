# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pypto.runtime import distributed_runner
from pypto_serving.model.deepseek_lowlat.worker_options import native_worker_options


def test_public_depth_forwarding_preserves_original_factory_validation_and_restores_scope(monkeypatch):
    import simpler.worker

    fake = Mock(return_value=object())
    monkeypatch.setattr(simpler.worker, "Worker", fake)
    original = distributed_runner._construct_worker
    with native_worker_options() as receipt:
        with pytest.raises(ValueError, match="positive finite"):
            distributed_runner._construct_worker(SimpleNamespace(device_ids=(7, 6)), "a2a3", "aicpu", 8,
                                                 startup_timeout_s=-1)
        distributed_runner._construct_worker(SimpleNamespace(device_ids=(7, 6)), "a2a3", "aicpu", 8,
                                             enable_sdma=True, startup_timeout_s=7200)
        options = fake.call_args.kwargs
        assert options == dict(level=3, device_ids=(7, 6), num_sub_workers=8, platform="a2a3",
                               runtime="aicpu", enable_sdma=True, startup_timeout_s=7200,
                               pipeline_depth=2, launch_depth=1)
        assert receipt["sdk_files_modified"] is False
    assert distributed_runner._construct_worker is original
    with pytest.raises(RuntimeError, match="verification"):
        with native_worker_options():
            raise RuntimeError("verification")
    assert distributed_runner._construct_worker is original


def test_unknown_factory_signature_fails_closed_before_constructing_worker(monkeypatch):
    fake = Mock()
    monkeypatch.setattr(distributed_runner, "_construct_worker", fake)
    with pytest.raises(ValueError, match="signature changed"):
        with native_worker_options():
            pytest.fail("unexpected factory accepted")
    fake.assert_not_called()
