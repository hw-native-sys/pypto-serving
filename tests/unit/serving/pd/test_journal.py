# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import stat

import pytest

from pypto_serving.serving.pd.config import PDConfig, PDRole
from pypto_serving.serving.pd.journal import DurablePDJournal
from pypto_serving.serving.pd.protocol import HandoffKey


def _config(path) -> PDConfig:
    return PDConfig(
        role=PDRole.PREFILL,
        node_id="p",
        run_id="run",
        control_host="127.0.0.1",
        control_port=29831,
        control_advertise_host="127.0.0.1",
        transfer_hostname="127.0.0.1",
        model_revision="model",
        journal_path=str(path),
    )


def test_journal_round_trip_is_durable_bounded_and_address_free(tmp_path) -> None:
    path = tmp_path / "pd.jsonl"
    config = _config(path)
    key = HandoffKey("request", "handoff", 1, 1, 1)
    journal = DurablePDJournal(str(path), config)
    journal.append("SERVICE_STARTING")
    journal.append("HANDOFF_CREATED", key=key, state="CREATED")
    journal.append(
        "HANDOFF_COMPLETED",
        key=key,
        reservation_id="reservation",
        manifest_hash="f" * 64,
        state="COMPLETED",
    )
    journal.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert records[1]["request_id"] == "request"
    serialized = path.read_text()
    assert "provider_envelope" not in serialized
    assert "native_address" not in serialized
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    audit = DurablePDJournal.audit(str(path), config)
    assert audit["records"] == 3
    assert audit["unresolved_handoffs"] == 0

    with pytest.raises(RuntimeError, match="completed control incarnation"):
        DurablePDJournal(str(path), config)


def test_journal_rejects_unresolved_handoff_after_restart(tmp_path) -> None:
    path = tmp_path / "pd.jsonl"
    config = _config(path)
    journal = DurablePDJournal(str(path), config)
    journal.append(
        "HANDOFF_CREATED",
        key=HandoffKey("request", "handoff", 1, 1, 1),
    )
    journal.close()

    with pytest.raises(RuntimeError, match="unresolved handoffs"):
        DurablePDJournal(str(path), config)


def test_journal_rejects_corruption_and_wrong_incarnation(tmp_path) -> None:
    path = tmp_path / "pd.jsonl"
    config = _config(path)
    journal = DurablePDJournal(str(path), config)
    journal.append("SERVICE_STARTING")
    journal.close()

    changed = path.read_text().replace("SERVICE_STARTING", "SERVICE_STARTED")
    path.write_text(changed)
    with pytest.raises(RuntimeError, match="hash chain"):
        DurablePDJournal(str(path), config)

    other_path = tmp_path / "other.jsonl"
    journal = DurablePDJournal(str(other_path), config)
    journal.append("SERVICE_STARTING")
    journal.close()
    wrong = PDConfig(
        **{
            **config.__dict__,
            "journal_path": str(other_path),
            "control_incarnation": 2,
        }
    )
    with pytest.raises(RuntimeError, match="identity"):
        DurablePDJournal(str(other_path), wrong)
