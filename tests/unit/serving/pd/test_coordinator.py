# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from pypto_serving.serving.pd.config import PDConfig, PDRole
from pypto_serving.serving.pd.coordinator import CoordinatorState, HandoffCoordinator
from pypto_serving.serving.pd.protocol import HandoffKey


def _config() -> PDConfig:
    return PDConfig(
        role=PDRole.PREFILL,
        node_id="p",
        run_id="run",
        control_host="127.0.0.1",
        control_port=29831,
        control_advertise_host="127.0.0.1",
        transfer_hostname="10.0.0.1",
        model_revision="model",
    )


def test_handoff_coordinator_is_idempotent_and_single_writer() -> None:
    coordinator = HandoffCoordinator(_config())
    key = HandoffKey("request", "handoff", 1, 1, 1)
    record = coordinator.register_handoff(
        key, prefill_node_id="p", decode_node_id="d"
    )
    assert record.prefill_node_id == "p"
    assert record.decode_node_id == "d"
    with pytest.raises(ValueError, match="already has"):
        coordinator.register_handoff(
            HandoffKey("request", "other", 1, 1, 1),
            prefill_node_id="p",
            decode_node_id="d",
        )
    assert coordinator.mark_reserved(record.key, "reservation").state is CoordinatorState.RESERVED
    assert coordinator.mark_reserved(record.key, "reservation").state is CoordinatorState.RESERVED
    assert coordinator.mark_ready(record.key, "manifest").state is CoordinatorState.READY
    assert coordinator.mark_ready(record.key, "manifest").state is CoordinatorState.READY
    assert coordinator.release(record.key).state is CoordinatorState.RELEASED
    assert coordinator.release(record.key).state is CoordinatorState.RELEASED
