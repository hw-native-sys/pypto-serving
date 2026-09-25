# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-side grammar providers; the device only receives packed allowed-token bits."""

from __future__ import annotations

import json
from typing import Protocol, Sequence

import numpy as np

from .spec import ConstraintSpec


class ConstraintState(Protocol):
    """A request's committed grammar position and speculative row planner."""

    def validate_draft_prefix(self, token_ids: Sequence[int]) -> int: ...

    def masks_for_rows(self, valid_draft_ids: Sequence[int]) -> np.ndarray: ...

    def accept(self, token_ids: Sequence[int]) -> None: ...

    def close(self) -> None: ...


class XGrammarProvider:
    """Compile model structural tags against the checkpoint's actual tokenizer."""

    def __init__(self, tokenizer) -> None:
        try:
            import xgrammar as xgr
        except ImportError as exc:
            raise ValueError("xgrammar is required for constrained tool generation") from exc

        backend_tokenizer = getattr(tokenizer, "tokenizer", None)
        if backend_tokenizer is None:
            raise ValueError("xgrammar requires a Hugging Face tokenizer backend")
        vocabulary = tokenizer.get_vocab()
        self.vocab_size = max(vocabulary.values()) + 1
        if self.vocab_size != len(vocabulary):
            raise ValueError("xgrammar requires a contiguous tokenizer vocabulary")
        self._xgr = xgr
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            backend_tokenizer, vocab_size=self.vocab_size
        )
        self._compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8, cache_enabled=True)

    def compile(self, spec: ConstraintSpec) -> "XGrammarState":
        if spec.provider_id != "xgrammar":
            raise ValueError(f"unsupported constraint provider {spec.provider_id!r}")
        structural_tag = self._xgr.get_model_structural_tag(
            spec.format_id,
            list(spec.tools),
            spec.tool_choice,
            reasoning=spec.reasoning,
            parallel_tool_calls=spec.parallel_tool_calls,
        )
        grammar = self._compiler.compile_structural_tag(
            json.dumps(structural_tag.model_dump(by_alias=True))
        )
        return XGrammarState(
            self._xgr.GrammarMatcher(grammar, max_rollback_tokens=7),
            self._xgr,
            self.vocab_size,
        )


class XGrammarState:
    """Advance only on actual outputs; simulate and roll back proposed drafts."""

    def __init__(self, matcher, xgr, vocab_size: int) -> None:
        self._matcher = matcher
        self._xgr = xgr
        self.vocab_size = vocab_size

    def validate_draft_prefix(self, token_ids: Sequence[int]) -> int:
        advanced = 0
        accepted = 0
        try:
            for token_id in token_ids:
                if self._matcher.is_terminated():
                    accepted += 1
                    continue
                if not self._matcher.accept_token(int(token_id)):
                    break
                advanced += 1
                accepted += 1
            return accepted
        finally:
            if advanced:
                self._matcher.rollback(advanced)

    def masks_for_rows(self, valid_draft_ids: Sequence[int]) -> np.ndarray:
        rows = len(valid_draft_ids) + 1
        bitmask = self._xgr.allocate_token_bitmask(rows, self.vocab_size)
        advanced = 0
        try:
            for row, token_id in enumerate(valid_draft_ids):
                if self._matcher.is_terminated():
                    bitmask[row].fill_(-1)
                    continue
                self._matcher.fill_next_token_bitmask(bitmask, row)
                if not self._matcher.accept_token(int(token_id)):
                    raise ValueError("a speculative token violates its grammar mask")
                advanced += 1
            if self._matcher.is_terminated():
                bitmask[-1].fill_(-1)
            else:
                self._matcher.fill_next_token_bitmask(bitmask, rows - 1)
            return bitmask.numpy().copy()
        finally:
            if advanced:
                self._matcher.rollback(advanced)

    def accept(self, token_ids: Sequence[int]) -> None:
        for token_id in token_ids:
            if self._matcher.is_terminated():
                return
            if not self._matcher.accept_token(int(token_id)):
                raise ValueError(f"generated token {token_id} violates its grammar")

    def close(self) -> None:
        self._matcher = None
