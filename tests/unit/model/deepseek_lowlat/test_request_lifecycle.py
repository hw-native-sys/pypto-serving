# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest

from pypto_serving.model.deepseek_lowlat.state import LowlatRequestState


def test_prefill_chunks_and_first_decode_cursor():
    state = LowlatRequestState("request", 1, 129)
    state.check_prefill(0, 128)
    state.commit_prefill(128)
    state.check_prefill(128, 1)
    state.commit_prefill(129)
    state.last_sampled_token = 42
    state.phase = "decode"
    state.check_decode(129, 42)
    assert state.kv_committed_tokens == 129
    with pytest.raises(ValueError, match="cursor/token"):
        state.check_decode(130, 42)
    with pytest.raises(ValueError, match="cursor/token"):
        state.check_decode(129, 43)


def test_failed_step_does_not_advance_by_validation():
    state = LowlatRequestState("request", 1, 128)
    state.check_prefill(0, 128)
    assert state.prefilled_tokens == state.kv_committed_tokens == 0
    with pytest.raises(ValueError, match="cursor"):
        state.check_prefill(128, 1)


def test_draining_state_cannot_accept_new_device_work():
    state = LowlatRequestState("cancelled", 7, 128, phase="draining")
    with pytest.raises(ValueError):
        state.check_prefill(0, 8)
    with pytest.raises(ValueError):
        state.check_decode(128, 42)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
