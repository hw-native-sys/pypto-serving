import json
from pathlib import Path

import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_CONTRACT
from pypto_serving.router.config import RouterConfig
from pypto_serving.serving.pd.config import PDRole, load_pd_document, resolve_pd_config
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
        model_contract=DSV4_DSPARK_K7_CONTRACT,
    )
    decode = resolve_pd_config(
        document,
        role=PDRole.DECODE,
        model_revision="dsv4",
        model_contract=DSV4_DSPARK_K7_CONTRACT,
    )
    router = RouterConfig.from_document(document)

    assert prefill.run_id == decode.run_id == router.run_id
    assert prefill.provider == router.provider == "mooncake"
    assert router.policy == "round_robin"
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
        model_contract=DSV4_DSPARK_K7_CONTRACT,
    )
    write_startup_record(
        config.log_dir,
        enabled=config.observability_enabled,
        values={"process": "prefill"},
    )
    assert Path(config.journal_path).parent.is_dir()
    assert not (Path(config.log_dir) / "startup.json").exists()
