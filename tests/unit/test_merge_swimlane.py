# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression coverage for the standalone swimlane merger's flow connections."""

import copy
import importlib.util
from pathlib import Path


_SCRIPT = (Path(__file__).resolve().parents[2] / ".agents/skills/profile-serving-simpler-swimlane"
           / "scripts/merge_swimlane.py")
_SPEC = importlib.util.spec_from_file_location("merge_swimlane", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _slice():
    return {"ph": "X", "name": "work", "pid": 1, "tid": 1, "ts": 100, "dur": 100}


def _flow(identifier, category="runtime", scope="global"):
    return [{"ph": phase, "id": identifier, "cat": category, "scope": scope,
             "pid": 1, "tid": 1, "ts": timestamp}
            for phase, timestamp in (("s", 110), ("t", 120), ("f", 130))]


def test_flow_ids_are_numeric_and_do_not_collide_with_serving():
    serving = {"traceEvents": [_slice(), *_flow(0), *_flow("1"), *_flow("0x2"),
                               dict(_slice(), bind_id=3)]}
    simpler = {"traceEvents": [_slice(), *_flow(7), *_flow(8)]}
    originals = copy.deepcopy((serving, simpler))

    result = _MODULE.merge(serving, simpler)["traceEvents"]
    assert result[:len(serving["traceEvents"])] == serving["traceEvents"]
    flows = [e for e in result[len(serving["traceEvents"]):] if e["ph"] in ("s", "t", "f")]
    assert len(flows) == 6
    assert all(type(e["id"]) is int and e["id"] not in {0, 1, 2, 3} for e in flows)
    assert len({e["id"] for e in flows}) == 2
    for identifier in {e["id"] for e in flows}:
        assert [e["ph"] for e in flows if e["id"] == identifier] == ["s", "t", "f"]
    assert (serving, simpler) == originals


def test_scoped_flows_remain_distinct_and_orphans_are_removed():
    serving = {"traceEvents": [_slice()]}
    simpler = {"traceEvents": [_slice(), *_flow(7, "dispatch"), *_flow(7, "copy"),
                               *_flow(7, "copy", "local"),
                               {"ph": "s", "id": 9, "pid": 1, "tid": 1, "ts": 140}]}
    result = _MODULE.merge(serving, simpler)["traceEvents"]
    flows = [e for e in result if e["ph"] in ("s", "t", "f")]
    assert len(flows) == 9
    assert len({e["id"] for e in flows}) == 3
    assert all(type(e["id"]) is int for e in flows)
