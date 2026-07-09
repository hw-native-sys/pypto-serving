# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


class TokenizerAdapter:
    """Minimal tokenizer interface required by the generation engine."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs without adding prompt specials."""
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs into text."""
        raise NotImplementedError

    @property
    def bos_token_id(self) -> int | None:
        """Return the beginning-of-sequence token ID, if available."""
        return None

    @property
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token ID, if available."""
        return None

    @property
    def pad_token_id(self) -> int | None:
        """Return the padding token ID, if available."""
        return None


@dataclass
class TransformersTokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter backed by ``transformers.AutoTokenizer``."""

    tokenizer: object

    @classmethod
    def from_pretrained(cls, model_dir: str, trust_remote_code: bool = False) -> "TransformersTokenizerAdapter":
        """Load a local Hugging Face tokenizer directory."""
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for the current local Hugging Face tokenizer adapter."
            ) from exc

        model_path = Path(model_dir)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning(
                "AutoTokenizer.from_pretrained failed for %s: %s; falling back to local tokenizer.json",
                model_path,
                exc,
            )
            tokenizer = _load_fast_tokenizer_from_file(model_path, PreTrainedTokenizerFast)
        return cls(tokenizer=tokenizer)

    @classmethod
    def from_tokenizer_file(cls, model_dir: str) -> "TransformersTokenizerAdapter":
        """Load ``tokenizer.json`` directly without consulting model config."""
        try:
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for the current local Hugging Face tokenizer adapter."
            ) from exc

        return cls(tokenizer=_load_fast_tokenizer_from_file(Path(model_dir), PreTrainedTokenizerFast))

    def encode(self, text: str) -> list[int]:
        """Encode text using the wrapped Hugging Face tokenizer."""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs, stripping special tokens (EOS / pad / im_end ...) from output."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    @property
    def bos_token_id(self) -> int | None:
        """Return the wrapped tokenizer BOS token ID."""
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        """Return the wrapped tokenizer EOS token ID."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """Return the wrapped tokenizer PAD token ID."""
        return self.tokenizer.pad_token_id


def _token_content(value: object) -> str | None:
    """Extract a special token string from tokenizer_config JSON."""
    if isinstance(value, dict):
        content = value.get("content")
        return content if isinstance(content, str) else None
    return value if isinstance(value, str) else None


def _load_fast_tokenizer_from_file(model_path: Path, tokenizer_cls: type) -> object:
    """Load a local tokenizer.json with special tokens from tokenizer_config."""
    tokenizer_file = model_path / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {model_path}")
    config_path = model_path / "tokenizer_config.json"
    tokenizer_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    special_tokens = {
        name: _token_content(tokenizer_config.get(name))
        for name in ("bos_token", "eos_token", "pad_token", "unk_token")
        if _token_content(tokenizer_config.get(name)) is not None
    }
    return tokenizer_cls(tokenizer_file=str(tokenizer_file), **special_tokens)
