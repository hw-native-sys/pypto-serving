# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Built-in model execution contracts and worker-side executor lookup."""

from __future__ import annotations

from dataclasses import dataclass

from pypto_serving.config.launch_options import FeatureOptions
from pypto_serving.model.integrations.base import ModelIntegration, Speculation
from pypto_serving.model.integrations.deepseek import DeepSeekV4Integration
from pypto_serving.model.integrations.deepseek_dspark import DeepSeekV4DSparkIntegration
from pypto_serving.model.integrations.qwen import QwenIntegration


_INTEGRATIONS: dict[str, ModelIntegration] = {
    integration.integration_id: integration
    for integration in (QwenIntegration(), DeepSeekV4Integration(), DeepSeekV4DSparkIntegration())
}


@dataclass(frozen=True)
class ModelFamily:
    name: str
    model_type: str
    architectures: frozenset[str]
    tokenizer_kind: str
    model_format: str
    integration_ids: dict[str, str]


_FAMILIES = (
    ModelFamily("qwen3", "qwen3", frozenset({"qwen3forcausallm"}), "transformers", "hf", {"": "qwen3_14b"}),
    ModelFamily(
        "deepseek_v4", "deepseek_v4", frozenset({"deepseekv4forcausallm"}),
        "deepseek_v4", "deepseek_v4_w8a8",
        {"": "deepseek_v4", "mtp": "deepseek_v4", "dspark": "deepseek_v4_dspark"},
    ),
)
_SUPPORTED_ARCHITECTURES = frozenset().union(*(family.architectures for family in _FAMILIES))


def select_model_family(metadata: dict[str, object]) -> ModelFamily:
    raw_type = metadata.get("model_type")
    raw_arches = metadata.get("architectures")
    if raw_type is not None and not isinstance(raw_type, str):
        raise ValueError(f"unsupported or ambiguous model architecture: model_type={raw_type!r}")
    if raw_arches is not None and (
        not isinstance(raw_arches, list) or not all(isinstance(arch, str) for arch in raw_arches)
    ):
        raise ValueError(f"unsupported or ambiguous model architecture: architectures={raw_arches!r}")

    model_type = raw_type.lower() if raw_type else ""
    architectures = {arch.lower() for arch in raw_arches or []}
    type_match = next((family for family in _FAMILIES if family.model_type == model_type), None)
    arch_matches = {family.name for family in _FAMILIES if architectures & family.architectures}
    if (
        len(arch_matches) > 1
        or bool(architectures - _SUPPORTED_ARCHITECTURES)
        or (model_type and type_match is None)
        or (type_match is not None and arch_matches and type_match.name not in arch_matches)
        or (type_match is None and not arch_matches)
    ):
        raise ValueError(
            "unsupported or ambiguous model architecture: "
            f"model_type={model_type!r}, architectures={sorted(architectures)}"
        )
    if type_match is not None:
        return type_match
    return next(family for family in _FAMILIES if family.name in arch_matches)


def family_for_integration(integration_id: str) -> ModelFamily:
    get_integration(integration_id)
    return next(family for family in _FAMILIES if integration_id in family.integration_ids.values())


def resolve_speculation(features: FeatureOptions) -> Speculation:
    config = features.speculative_config
    alias = features.num_speculative_tokens
    if config is not None:
        if alias is not None:
            raise ValueError("--speculative-config cannot be combined with --num-speculative-tokens")
        method = config.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("--speculative-config requires a non-empty string method")
        explicit_tokens = "num_speculative_tokens" in config
        tokens = config.get("num_speculative_tokens", 0)
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            raise ValueError("num_speculative_tokens must be an integer")
        if tokens < 0:
            raise ValueError("num_speculative_tokens must be non-negative")
        return Speculation(method, tokens, explicit_tokens)
    if alias is None:
        return Speculation()
    if alias < 0:
        raise ValueError("--num-speculative-tokens must be non-negative")
    return Speculation("", alias, True)


def select_integration(metadata: dict[str, object], speculation: Speculation) -> ModelIntegration:
    family = select_model_family(metadata)
    key = family.integration_ids.get(speculation.method)
    if key is None:
        raise ValueError(f"{family.name} does not support speculative method {speculation.method!r}")
    return _INTEGRATIONS[key]


def get_integration(integration_id: str) -> ModelIntegration:
    try:
        return _INTEGRATIONS[integration_id]
    except KeyError as exc:
        raise ValueError(f"unsupported model integration: {integration_id!r}") from exc
