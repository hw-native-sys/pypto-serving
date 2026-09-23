# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Schema-derived token constraints for DeepSeek V4 tool calls."""

from __future__ import annotations

import copy
import json
from functools import lru_cache

import torch


def _xgrammar():
    try:
        import xgrammar
    except ImportError as exc:
        raise ValueError("Tool schema constraints require xgrammar>=0.2.7; install serving runtime dependencies") from exc
    return xgrammar


@lru_cache(maxsize=128)
def _validate_grammar(grammar: str) -> str:
    try:
        _xgrammar().Grammar.from_structural_tag(grammar)
    except RuntimeError as exc:
        raise ValueError(f"Unsupported tool schema: {exc}") from exc
    return grammar


def _relocate_refs(node, prefix: str) -> None:
    if isinstance(node, dict):
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#"):
            node["$ref"] = prefix + reference[1:]
        for key, value in node.items():
            if key not in {"const", "enum", "default", "examples"}:
                _relocate_refs(value, prefix)
    elif isinstance(node, list):
        for value in node:
            _relocate_refs(value, prefix)


def _strict_parameters(schema: dict) -> dict:
    """Keep DSML's string attribute consistent with the JSON value type.

    XGrammar 0.2.7's deepseek_xml permits either string attribute regardless
    of the property's type. Use raw text only for unconstrained strings, and
    standard JSON Schema for other values, including constrained strings.
    """
    supported = {"type", "properties", "required", "additionalProperties", "$defs", "definitions",
                 "title", "description", "$schema", "default", "examples"}
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise ValueError("Strict function parameters must be an object schema")
    if set(schema) - supported:
        raise ValueError(f"Unsupported strict tool root schema keywords: {sorted(set(schema) - supported)}")
    properties = schema.get("properties", {})
    required_names = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required_names, list) or not all(
        isinstance(name, str) for name in required_names
    ):
        raise ValueError("Tool schemas need a properties object and a list of required parameter names")
    required = set(required_names)
    if not required.issubset(properties):
        raise ValueError("Required tool parameters must be declared in properties")
    elements = []
    annotations = {"type", "title", "description", "default", "examples"}
    for name, value in properties.items():
        if not name or any(char in name for char in '\"<>\n\r'):
            raise ValueError("Tool parameter names must be representable as DSML attributes")
        raw_string = isinstance(value, dict) and value.get("type") == "string" and not set(value) - annotations
        if raw_string:
            content = {"type": "any_text", "excludes": ["</｜DSML｜parameter>"]}
        else:
            # Preserve local references to sibling properties and definitions by
            # retaining the complete root schema under a private definition.
            root_name = "__pypto_tool_root"
            relocated = copy.deepcopy(schema)
            _relocate_refs(relocated, f"#/$defs/{root_name}")
            pointer = name.replace("~", "~0").replace("/", "~1")
            value_schema = {"$ref": f"#/$defs/{root_name}/properties/{pointer}",
                            "$defs": {root_name: relocated}}
            content = {"type": "json_schema", "json_schema": value_schema}
        tag = {"type": "tag", "begin": f'<｜DSML｜parameter name="{name}" string="{str(raw_string).lower()}">',
               "content": content, "end": "</｜DSML｜parameter>\n"}
        elements.append(tag if name in required else {"type": "optional", "content": tag})
    return {"type": "sequence", "elements": elements} if elements else {"type": "const_string", "value": ""}


def _apply_strict_parameters(node, schemas: dict[str, dict]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "tag" and node.get("begin") in schemas:
            node["content"] = _strict_parameters(schemas[node["begin"]])
        else:
            for value in node.values():
                _apply_strict_parameters(value, schemas)
    elif isinstance(node, list):
        for item in node:
            _apply_strict_parameters(item, schemas)


def tool_grammar(tools: list[dict], choice: str | dict, *, thinking: bool,
                 parallel: bool, enforce: bool = False) -> str | None:
    """Build a grammar from request schemas, without rewriting tool arguments."""
    if not tools or choice == "none":
        return None
    if choice == "auto" and not enforce and not any(t["function"].get("strict") for t in tools):
        return None
    tools = copy.deepcopy(tools)
    if enforce:
        for tool in tools:
            tool["function"]["strict"] = True
    xgr = _xgrammar()
    tag = xgr.get_model_structural_tag(
        "deepseek_v4", tools=tools, tool_choice=choice,
        reasoning=thinking, parallel_tool_calls=parallel,
    )
    document = tag.model_dump()
    schemas = {f'<｜DSML｜invoke name="{tool["function"]["name"]}">\n':
               tool["function"].get("parameters", {"type": "object", "properties": {}})
               for tool in tools if tool["function"].get("strict") is not False}
    _apply_strict_parameters(document, schemas)
    return _validate_grammar(json.dumps(document, ensure_ascii=False))


class ConstraintCompiler:
    """One tokenizer/compiler cache per worker; matchers belong to requests."""

    def __init__(self, tokenizer, vocab_size: int):
        self.xgr = _xgrammar()
        self.vocab_size = vocab_size
        self.stop_id = tokenizer.eos_token_id
        if self.stop_id is None:
            raise ValueError("Tool constraints require a tokenizer EOS token")
        info = self.xgr.TokenizerInfo.from_huggingface(
            tokenizer.tokenizer, vocab_size=vocab_size, stop_token_ids=[self.stop_id],
        )
        self.compiler = self.xgr.GrammarCompiler(info, max_threads=2, cache_limit_bytes=128 << 20)

    def create(self, grammar: str):
        compiled = self.compiler.compile_structural_tag(grammar)
        return TokenConstraint(self.xgr.GrammarMatcher(compiled), self.vocab_size, self.stop_id)


class TokenConstraint:
    """Mutable grammar state, with rollback when previewing speculative drafts."""

    def __init__(self, matcher, vocab_size: int, stop_id: int):
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.stop_id = stop_id

    def bitmask(self) -> torch.Tensor:
        mask = torch.full((1, (self.vocab_size + 31) // 32), -1, dtype=torch.int32)
        if self.matcher.is_terminated():
            mask.zero_()
            word, bit = divmod(self.stop_id, 32)
            value = 1 << bit
            mask[0, word] = value if value < (1 << 31) else value - (1 << 32)
        else:
            self.matcher.fill_next_token_bitmask(mask)
        return mask[0]

    def accept(self, tokens) -> None:
        for token in tokens:
            token = int(token)
            if self.matcher.is_terminated() and token == self.stop_id:
                continue
            if not self.matcher.accept_token(token):
                raise RuntimeError(f"Sampler emitted token {token} outside the tool schema grammar")

    def preview(self, draft_tokens: list[int], width: int) -> torch.Tensor:
        """Mask each target row under the preceding draft prefix, then roll back.

        Rows beyond an invalid draft are unreachable: acceptance must stop at
        the preceding target row, whose mask excludes that invalid proposal.
        """
        result = torch.full((width, (self.vocab_size + 31) // 32), -1, dtype=torch.int32)
        advanced = 0
        try:
            for row in range(width):
                result[row] = self.bitmask()
                if row >= len(draft_tokens) or row + 1 == width:
                    break
                token = int(draft_tokens[row])
                if self.matcher.is_terminated():
                    if token != self.stop_id:
                        break
                    continue
                if not self.matcher.accept_token(token):
                    break
                advanced += 1
        finally:
            if advanced:
                self.matcher.rollback(advanced)
        return result

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float().clone()
        mask = self.bitmask().to(logits.device)
        ids = torch.arange(logits.numel(), device=logits.device)
        allowed = ((mask[ids // 32] >> (ids % 32)) & 1).bool()
        logits.masked_fill_(~allowed, -float("inf"))
        if not torch.isfinite(logits).any():
            raise RuntimeError("No finite logits are allowed by the tool schema grammar")
        return logits
