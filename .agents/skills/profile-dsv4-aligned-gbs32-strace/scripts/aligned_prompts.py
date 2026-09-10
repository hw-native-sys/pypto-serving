# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load and validate the ordered 32-attraction alignment inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REQUEST_COUNT = 32
PROMPT_TOKENS = 64
PROMPT_MATRIX_SHA256 = "346e69fa38fc12f9412564288a16b8a674a47365fa8488c690ff988c9c3fdf95"
PROMPTS_PATH = Path(__file__).resolve().parents[1] / "assets" / "prompts.json"
CHAT_PREFIX = "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>"
CHAT_SUFFIX = "<\uff5cAssistant\uff5c></think>"


def token_sha256(token_ids: list) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_prompt_records() -> list[dict]:
    records = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    if len(records) != REQUEST_COUNT or [row["index"] for row in records] != list(range(REQUEST_COUNT)):
        raise ValueError("aligned prompts must have ordered indices 0 through 31")
    for key in ("prompt", "attraction"):
        if len({row[key] for row in records}) != REQUEST_COUNT:
            raise ValueError(f"aligned prompts must have 32 unique {key} values")
    if any(row["input_tokens"] != PROMPT_TOKENS for row in records):
        raise ValueError("aligned prompts must specify 64 input tokens each")
    return records


def validate_prompt_manifest(manifest: list[dict]) -> str:
    records = load_prompt_records()
    if len(manifest) != REQUEST_COUNT:
        raise ValueError("aligned prompt manifest must contain 32 requests")
    matrix = []
    for expected, actual in zip(records, manifest):
        index = expected["index"]
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"aligned prompt {index} metadata/order mismatch")
        if actual.get("rendered_prompt") != CHAT_PREFIX + expected["prompt"] + CHAT_SUFFIX:
            raise ValueError(f"aligned prompt {index} chat rendering mismatch")
        ids = actual.get("token_ids")
        if not isinstance(ids, list) or len(ids) != PROMPT_TOKENS:
            raise ValueError(f"aligned prompt {index} must contain 64 token IDs")
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f"aligned prompt {index} has invalid token IDs")
        matrix.append(ids)
    if len({tuple(ids) for ids in matrix}) != REQUEST_COUNT:
        raise ValueError("aligned prompt token sequences must be unique")
    for record, ids in zip(records, matrix):
        if token_sha256(ids)[:12] != record["token_sha256_12"]:
            raise ValueError(f"aligned prompt {record['index']} token hash mismatch")
    digest = token_sha256(matrix)
    if digest != PROMPT_MATRIX_SHA256:
        raise ValueError(f"aligned prompt token matrix hash mismatch: {digest}")
    return digest


def build_prompt_manifest(tokenizer) -> list[dict]:
    manifest = []
    for record in load_prompt_records():
        rendered = CHAT_PREFIX + record["prompt"] + CHAT_SUFFIX
        manifest.append({**record, "rendered_prompt": rendered, "token_ids": list(tokenizer.encode(rendered))})
    validate_prompt_manifest(manifest)
    return manifest
