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

from pypto_serving.model.model_family import detect_model_family, read_model_config


logger = logging.getLogger(__name__)


class TokenizerAdapter:
    """Minimal tokenizer interface required by the generation engine."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs without adding prompt specials."""
        raise NotImplementedError

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs into text."""
        raise NotImplementedError

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary used to resolve model control tokens."""
        raise NotImplementedError

    @property
    def all_special_ids(self) -> tuple[int, ...]:
        """Return token IDs that must not leak into public text."""
        return ()

    @property
    def output_parser_id(self) -> str | None:
        """Return the Serving output parser registered for this tokenizer."""
        return None

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs) -> str:
        """Encode chat messages into the model's generation prompt."""
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

        fallback_errors: tuple[type[BaseException], ...] = (OSError, ValueError, TypeError, AttributeError)
        try:
            from huggingface_hub.errors import StrictDataclassFieldValidationError
        except ImportError:
            pass
        else:
            fallback_errors += (StrictDataclassFieldValidationError,)

        model_path = Path(model_dir)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
        except fallback_errors as exc:
            if not (model_path / "tokenizer.json").exists():
                raise
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

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs with request-selectable special-token handling."""
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def get_vocab(self) -> dict[str, int]:
        """Return a stable copy of the wrapped tokenizer vocabulary."""
        return dict(self.tokenizer.get_vocab())

    @property
    def all_special_ids(self) -> tuple[int, ...]:
        """Return all special-token IDs exposed by the wrapped tokenizer."""
        return tuple(int(token_id) for token_id in self.tokenizer.all_special_ids)

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs) -> str:
        """Apply the Hugging Face chat template loaded with the tokenizer."""
        return self.tokenizer.apply_chat_template(messages, **kwargs)

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


@dataclass
class DeepSeekV4TokenizerAdapter(TransformersTokenizerAdapter):
    """Tokenizer adapter for DeepSeek V4's Python-defined chat encoding."""

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs) -> str:
        from pypto_serving.model.deepseek.encoding import encode_messages

        thinking = bool(kwargs.get("thinking", False) or kwargs.get("enable_thinking", False))
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort == "none":
            thinking = False
            reasoning_effort = None
        return encode_messages(
            messages,
            thinking=thinking,
            reasoning_effort=reasoning_effort if isinstance(reasoning_effort, str) else None,
            tools=kwargs.get("tools"),
            drop_thinking=kwargs.get("drop_thinking", True),
        )

    @property
    def output_parser_id(self) -> str | None:
        """DeepSeek V4 emits reasoning control tokens understood by Serving."""
        return "deepseek_v4"


def load_tokenizer(model_dir: str | Path, *, trust_remote_code: bool = False) -> TokenizerAdapter:
    """Load a local tokenizer and select any model-specific chat encoding."""
    model_path = Path(model_dir)
    family = detect_model_family(read_model_config(model_path))
    if family == "deepseek_v41":
        from .deepseek_v41.tokenizer import DeepSeekV41TokenizerAdapter

        adapter_cls = DeepSeekV41TokenizerAdapter
    elif family == "deepseek_v4":
        adapter_cls = DeepSeekV4TokenizerAdapter
    else:
        adapter_cls = TransformersTokenizerAdapter
    if (model_path / "tokenizer.json").exists():
        return adapter_cls.from_tokenizer_file(str(model_path))
    return adapter_cls.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
    )

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
    tokenizer_payload = json.loads(tokenizer_file.read_text())
    config_path = model_path / "tokenizer_config.json"
    tokenizer_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    special_tokens: dict[str, object] = {
        name: _token_content(tokenizer_config.get(name))
        for name in ("bos_token", "eos_token", "pad_token", "unk_token")
        if _token_content(tokenizer_config.get(name)) is not None
    }
    registered = set(special_tokens.values())
    additional_special_tokens: list[str] = []
    for token in tokenizer_payload.get("added_tokens", ()):
        if not isinstance(token, dict) or token.get("special") is not True:
            continue
        content = token.get("content")
        if (
            isinstance(content, str)
            and content not in registered
        ):
            additional_special_tokens.append(content)
            registered.add(content)
    if additional_special_tokens:
        special_tokens["additional_special_tokens"] = additional_special_tokens
    chat_template = tokenizer_config.get("chat_template")
    if isinstance(chat_template, (str, dict)):
        special_tokens["chat_template"] = chat_template
    return tokenizer_cls(tokenizer_file=str(tokenizer_file), **special_tokens)
