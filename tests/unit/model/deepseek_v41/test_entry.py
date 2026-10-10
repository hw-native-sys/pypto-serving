# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Metadata and tokenizer entry tests through the public serving interfaces."""

import copy
import json
from pathlib import Path

import pytest

from pypto_serving.model.deepseek_v41.config import V41TextConfig, load_text_config
from pypto_serving.model.model_family import detect_model_family
from pypto_serving.model.model_loader import ModelLoader
from pypto_serving.model.tokenizer import load_tokenizer


FIXTURE = Path(__file__).resolve().parents[4] / "tests/fixtures/deepseek_v41/config.json"


@pytest.fixture
def raw():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def model_dir(tmp_path, raw):
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("metadata", [
    {"model_type": "deepseek_v41"},
    {"architectures": ["DeepseekV41ForCausalLM"]},
    {"model_type": "deepseek_v4", "architectures": ["DeepseekV41ForCausalLM"]},
])
def test_v41_detection_precedes_other_families(metadata):
    assert detect_model_family(metadata) == "deepseek_v41"


@pytest.mark.parametrize("metadata,expected", [
    ({"model_type": "deepseek_v4"}, "deepseek_v4"),
    ({"model_type": "qwen3"}, "qwen"),
    ({"architectures": None}, "qwen"),
])
def test_existing_family_detection(metadata, expected):
    assert detect_model_family(metadata) == expected


def test_metadata_only_read_ignores_optional_modules(model_dir, raw):
    before = copy.deepcopy(raw)
    config = load_text_config(model_dir)
    assert (config.hidden_size, config.num_hidden_layers, config.hc_mult) == (5120, 40, 4)
    assert len(config.compress_ratios) == 40
    assert set(config.compress_ratios) == {0, 1, 2}
    assert (config.bos_token_id, config.eos_token_id, config.pad_token_id) == (0, 1, 2)
    raw["vision_config"] = {"unimplemented": True}
    raw["text_config"]["engram_config"] = {"unimplemented": True}
    assert V41TextConfig.from_dict(raw) == config
    assert V41TextConfig.from_dict(before) == config
    assert list(model_dir.iterdir()) == [model_dir / "config.json"]


@pytest.mark.parametrize("field,value", [
    ("hidden_size", True), ("num_hidden_layers", 0), ("hc_mult", -1),
    ("rms_norm_eps", float("nan")), ("rms_norm_eps", 0),
    ("compress_ratios", [0]), ("compress_ratios", [3] * 40),
    ("compress_ratios", [True] * 40), ("model_type", "deepseek_v4"),
])
def test_invalid_text_config(raw, field, value):
    raw["text_config"][field] = value
    with pytest.raises(ValueError):
        V41TextConfig.from_dict(raw)


@pytest.mark.parametrize("value", [-1, True, 129280])
def test_invalid_special_id(raw, value):
    raw["eos_token_id"] = value
    with pytest.raises(ValueError, match="eos_token_id"):
        V41TextConfig.from_dict(raw)


@pytest.mark.parametrize("model_format", ["hf", "deepseek_v4"])
def test_loading_never_falls_through_to_qwen(model_dir, model_format):
    with pytest.raises(ValueError, match="requires model_format='deepseek_v41'"):
        ModelLoader().load("v41", str(model_dir), model_format=model_format)


def test_cli_rejects_execution_before_device_setup(model_dir):
    from pypto_serving.cli.main import build_parser, build_serving_engine_config

    args = build_parser().parse_args([
        "--model", str(model_dir), "--platform", "a5", "--tp", "4", "--dp", "2", "--ep", "8",
        "--devices", "0,1,2,3,4,5,6,7",
    ])
    with pytest.raises(NotImplementedError, match="verified lib composite bindings"):
        build_serving_engine_config(args)


def test_local_tokenizer_round_trip_and_chat(model_dir):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from pypto_serving.model.deepseek_v41.encoding import (
        BOS_TOKEN, EOS_TOKEN, USER_TOKEN, ASSISTANT_TOKEN,
    )

    special = [BOS_TOKEN, EOS_TOKEN, "<pad>", USER_TOKEN, ASSISTANT_TOKEN, "</think>"]
    vocab = {token: i for i, token in enumerate(special + ["[UNK]", "hello", "world"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(special)
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": BOS_TOKEN, "eos_token": EOS_TOKEN, "pad_token": "<pad>",
    }), encoding="utf-8")
    adapter = load_tokenizer(model_dir)
    assert adapter.encode("hello world") == [vocab["hello"], vocab["world"]]
    assert adapter.decode(adapter.encode("hello world")) == "hello world"
    ids = adapter.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=True)
    assert ids == [vocab[BOS_TOKEN], vocab[USER_TOKEN], vocab["hello"],
                   vocab[ASSISTANT_TOKEN], vocab["</think>"]]
    assert adapter.bos_token_id == 0 and adapter.eos_token_id == 1
    assert adapter.pad_token_id == 2
    assert adapter.decode(ids) == "hello"
