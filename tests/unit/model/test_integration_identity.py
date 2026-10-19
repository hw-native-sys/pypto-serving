# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from pypto_serving.model.integrations import (
    family_for_integration, select_integration, select_model_family,
)
from pypto_serving.model.integrations.base import Speculation
from pypto_serving.model.metadata import read_model_metadata
from pypto_serving.model.tokenizer import (
    DeepSeekV4TokenizerAdapter, TransformersTokenizerAdapter, load_tokenizer,
)
from pypto_serving.serving.server import serving_worker


@pytest.mark.parametrize("metadata,expected", [
    ({"model_type": "qwen3"}, "qwen3"),
    ({"architectures": ["Qwen3ForCausalLM"]}, "qwen3"),
    ({"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]}, "deepseek_v4"),
])
def test_model_family_resolves_supported_checkpoint_metadata(metadata, expected):
    assert select_model_family(metadata).name == expected


@pytest.mark.parametrize("metadata", [
    {},
    {"model_type": "unknown"},
    {"model_type": "qwen3", "architectures": ["UnknownForCausalLM"]},
    {"model_type": "qwen3", "architectures": ["DeepseekV4ForCausalLM"]},
    {"architectures": ["Qwen3ForCausalLM", "DeepseekV4ForCausalLM"]},
    {"model_type": "qwen3", "architectures": "Qwen3ForCausalLM"},
])
def test_model_family_rejects_unknown_or_conflicting_metadata(metadata):
    with pytest.raises(ValueError, match="unsupported or ambiguous model architecture"):
        select_model_family(metadata)


@pytest.mark.parametrize("method,expected", [
    ("", "deepseek_v4"), ("mtp", "deepseek_v4"), ("dspark", "deepseek_v4_dspark"),
])
def test_deepseek_variants_share_one_family(method, expected):
    integration = select_integration({"model_type": "deepseek_v4"}, Speculation(method))
    assert integration.integration_id == expected
    assert family_for_integration(expected).name == "deepseek_v4"


@pytest.mark.parametrize("contents,error", [
    (None, "missing model config"),
    ("{", "cannot read model config"),
    ("[]", "model config must be a JSON object"),
])
def test_model_metadata_reader_reports_invalid_config(tmp_path, contents, error):
    if contents is not None:
        (tmp_path / "config.json").write_text(contents)
    with pytest.raises(ValueError, match=error):
        read_model_metadata(tmp_path)


def test_tokenizer_uses_resolved_family_and_keeps_standalone_generic_path(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        TransformersTokenizerAdapter, "from_tokenizer_file",
        classmethod(lambda cls, _: cls),
    )

    assert load_tokenizer(tmp_path, integration_id="deepseek_v4_dspark") is DeepSeekV4TokenizerAdapter
    assert load_tokenizer(tmp_path) is DeepSeekV4TokenizerAdapter
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown"}))
    assert load_tokenizer(tmp_path) is TransformersTokenizerAdapter
    with pytest.raises(ValueError, match="unsupported model integration"):
        load_tokenizer(tmp_path, integration_id="missing")


def _worker_for_model(tmp_path, monkeypatch, model_type, integration_id):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    worker = serving_worker.WorkerProcess.__new__(serving_worker.WorkerProcess)
    worker.config = SimpleNamespace(
        worker_device_ids=lambda: (0,), profile_config=None,
        model_id="model", model_dir=str(tmp_path), integration_id=integration_id,
        device_id=0, dp_rank=0, platform="a2a3", executor_kwargs={}, runtime_config=None,
    )
    worker._configure_pypto_build_dir = lambda _: None
    monkeypatch.setattr(serving_worker, "configure_profiler", lambda *args, **kwargs: None)
    monkeypatch.setattr(serving_worker, "profile_span", lambda *args, **kwargs: nullcontext())
    return worker


def test_worker_rejects_mismatched_checkpoint_before_executor_import(tmp_path, monkeypatch):
    worker = _worker_for_model(tmp_path, monkeypatch, "qwen3", "deepseek_v4_dspark")
    integration = select_integration({"model_type": "deepseek_v4"}, Speculation("dspark"))
    monkeypatch.setattr(integration, "executor_class", lambda: pytest.fail("executor imported"))

    with pytest.raises(ValueError, match="expects deepseek_v4, but checkpoint declares qwen3"):
        worker.init_device_and_model()


@pytest.mark.parametrize("model_type,integration_id,model_format", [
    ("qwen3", "qwen3_14b", "hf"),
    ("deepseek_v4", "deepseek_v4_dspark", "deepseek_v4_w8a8"),
])
def test_worker_passes_selected_format_to_loader(
    tmp_path, monkeypatch, model_type, integration_id, model_format,
):
    worker = _worker_for_model(tmp_path, monkeypatch, model_type, integration_id)
    integration = select_integration(
        {"model_type": model_type},
        Speculation("dspark" if integration_id == "deepseek_v4_dspark" else ""),
    )
    calls = []

    class Executor:
        def __init__(self, **kwargs):
            pass

        def register_model(self, model_id, record):
            return 3

    def load(self, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            config=object(), tokenizer=object(), layer_specs=[],
            runtime_model=SimpleNamespace(runtime=SimpleNamespace(page_size=32)),
        )

    monkeypatch.setattr(integration, "executor_class", lambda: Executor)
    monkeypatch.setattr("pypto_serving.model.model_loader.ModelLoader.load", load)

    assert worker.init_device_and_model() == 3
    assert calls[0]["integration_id"] == integration_id
    assert calls[0]["model_format"] == model_format
