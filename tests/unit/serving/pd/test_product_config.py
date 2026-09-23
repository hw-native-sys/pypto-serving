# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import json
from pathlib import Path

import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_ADAPTER
from pypto_serving.router.config import RouterConfig
from pypto_serving.serving.pd.config import (
    PDPrefixCacheMode,
    PDRole,
    load_pd_document,
    resolve_pd_config,
)
from pypto_serving.serving.pd.observability import write_startup_record


def _write_config(tmp_path: Path, extra: dict | None = None) -> Path:
    value = {
        "runtime": {
            "prefill": [{"host": "10.0.0.1", "port": 8111}],
            "decode": [{"host": "10.0.0.2", "port": 8111}],
        },
        "observability": {"root": str(tmp_path / "logs")},
    }
    if extra:
        value.update(extra)
    path = tmp_path / "pd.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_minimal_config_resolves_shared_identity_and_default_paths(tmp_path) -> None:
    document = load_pd_document(_write_config(tmp_path))
    prefill = resolve_pd_config(
        document,
        role=PDRole.PREFILL,
        model_revision="dsv4",
        model_adapter=DSV4_DSPARK_K7_ADAPTER,
    )
    decode = resolve_pd_config(
        document,
        role=PDRole.DECODE,
        model_revision="dsv4",
        model_adapter=DSV4_DSPARK_K7_ADAPTER,
    )
    router = RouterConfig.from_document(document)

    assert prefill.run_id == decode.run_id == router.run_id
    assert prefill.provider == router.provider == "mooncake"
    assert router.policy == "round_robin"
    assert document.runtime.enable_chunk_overlap
    assert prefill.enable_chunk_overlap and decode.enable_chunk_overlap
    assert prefill.node_id.startswith("prefill-")
    assert decode.node_id.startswith("decode-")
    assert Path(prefill.journal_path).parent.is_dir()
    assert Path(router.journal_path).parent.is_dir()


def test_config_rejects_unknown_fields(tmp_path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["runtime"]["fallback"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown runtime fields"):
        load_pd_document(path)


def test_config_rejects_unimplemented_transfer_provider(tmp_path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["runtime"]["provider"] = "simpler"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported runtime.provider 'simpler'"):
        load_pd_document(path)


def test_observability_can_be_disabled_without_disabling_state(tmp_path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["observability"]["enabled"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    document = load_pd_document(path)
    config = resolve_pd_config(
        document,
        role=PDRole.PREFILL,
        model_revision="dsv4",
        model_adapter=DSV4_DSPARK_K7_ADAPTER,
    )
    write_startup_record(
        config.log_dir,
        enabled=config.observability_enabled,
        values={"process": "prefill"},
    )
    assert Path(config.journal_path).parent.is_dir()
    assert not (Path(config.log_dir) / "startup.json").exists()


@pytest.mark.parametrize("overlap", (False, True))
def test_runtime_controls_are_shared_by_router_and_nodes(tmp_path, overlap) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["runtime"].update(
        {
            "generation": 3,
            "route_epoch": 5,
            "control_incarnation": 7,
            "max_active_handoffs": 2,
            "max_pending_handoffs": 6,
            "enable_chunk_overlap": overlap,
        }
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    document = load_pd_document(path)
    node = resolve_pd_config(
        document,
        role=PDRole.PREFILL,
        model_revision="dsv4",
        model_adapter=DSV4_DSPARK_K7_ADAPTER,
    )
    router = RouterConfig.from_document(document)

    assert node.generation == router.data_generation == 3
    assert node.route_epoch == router.route_epoch == 5
    assert node.control_incarnation == router.control_incarnation == 7
    assert node.max_active_handoffs == router.max_active_handoffs == 2
    assert node.max_pending_handoffs == router.max_pending_handoffs == 6
    assert node.enable_chunk_overlap is overlap


@pytest.mark.parametrize(
    ("mode", "prefill_enabled", "decode_enabled"),
    (
        ("disabled", False, False),
        ("d_only", False, True),
        ("independent", True, True),
    ),
)
def test_prefix_cache_profile_is_role_specific(
    tmp_path, mode, prefill_enabled, decode_enabled
) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["runtime"]["prefix_cache_mode"] = mode
    path.write_text(json.dumps(value), encoding="utf-8")
    document = load_pd_document(path)
    assert document.runtime.prefix_cache_mode is PDPrefixCacheMode(mode)
    prefill = resolve_pd_config(
        document,
        role=PDRole.PREFILL,
        model_revision="dsv4",
        model_adapter=DSV4_DSPARK_K7_ADAPTER,
    )
    decode = resolve_pd_config(
        document,
        role=PDRole.DECODE,
        model_revision="dsv4",
        model_adapter=DSV4_DSPARK_K7_ADAPTER,
    )
    assert prefill.prefix_cache_enabled is prefill_enabled
    assert decode.prefix_cache_enabled is decode_enabled


def test_profile_enables_independent_p_and_d_caches(tmp_path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["runtime"]["prefix_cache_mode"] = "independent"
    path.write_text(json.dumps(value), encoding="utf-8")
    document = load_pd_document(path)
    resolved = tuple(
        resolve_pd_config(
            document,
            role=role,
            model_revision="dsv4",
            model_adapter=DSV4_DSPARK_K7_ADAPTER,
        )
        for role in (PDRole.PREFILL, PDRole.DECODE)
    )
    assert all(config.prefix_cache_enabled for config in resolved)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("generation", 0, "runtime.generation"),
        ("route_epoch", 0, "runtime.route_epoch"),
        ("control_incarnation", 0, "runtime.control_incarnation"),
        ("max_active_handoffs", 0, "runtime.max_active_handoffs"),
        ("max_pending_handoffs", 0, "runtime.max_pending_handoffs"),
        ("enable_chunk_overlap", 1, "runtime.enable_chunk_overlap"),
    ),
)
def test_runtime_controls_are_strict(tmp_path, field, value, message) -> None:
    path = _write_config(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["runtime"][field] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_pd_document(path)
