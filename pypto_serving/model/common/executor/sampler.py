# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import warnings

import torch

from pypto_serving.config.types import GenerateConfig, SamplingCandidates, SamplingParams


class Sampler:
    """Token sampler that supports greedy, top-k, and top-p sampling."""

    def sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        """Sample one token ID from logits using the supplied sampling params."""
        logits = self._sanitize_logits(logits)
        if params.temperature <= 0.0:
            return self._greedy_token(logits)

        scaled = logits / max(params.temperature, 1e-5)

        if params.top_k is not None and params.top_k > 0 and params.top_k < scaled.numel():
            topk_values, topk_indices = torch.topk(scaled, params.top_k)
            filtered = torch.full_like(scaled, float("-inf"))
            filtered[topk_indices] = topk_values
            scaled = filtered

        probs = torch.softmax(scaled, dim=-1)
        if not self._is_valid_distribution(probs):
            return self._greedy_token(logits)

        if 0.0 < params.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= params.top_p
            keep[0] = True
            filtered_probs = torch.zeros_like(probs)
            filtered_probs[sorted_indices[keep]] = probs[sorted_indices[keep]]
            total = filtered_probs.sum()
            if not torch.isfinite(total) or total.item() <= 0.0:
                return self._greedy_token(logits)
            probs = filtered_probs / total
            if not self._is_valid_distribution(probs):
                return self._greedy_token(logits)

        token = torch.multinomial(
            probs,
            num_samples=1,
            generator=self._generator(probs, params),
        )
        return int(token.item())

    def sample_from_candidates(
        self, candidates: SamplingCandidates, row_idx: int, params: SamplingParams
    ) -> int:
        """Sample from an executor-provided top-k candidate row."""
        value_rows = candidates.values.shape[0]
        token_id_rows = candidates.token_ids.shape[0]
        if value_rows != token_id_rows:
            raise ValueError(f"candidate values/ids row mismatch: {value_rows} != {token_id_rows}")
        if row_idx < 0 or row_idx >= value_rows:
            raise ValueError(f"row_idx {row_idx} is out of bounds for candidates with {value_rows} rows")

        values = candidates.values[row_idx].float()
        token_ids = candidates.token_ids[row_idx].long()
        if values.numel() == 0 or token_ids.numel() == 0:
            raise ValueError("sampling candidates must not be empty")
        if values.numel() != token_ids.numel():
            raise ValueError(
                f"candidate values/ids width mismatch: {values.numel()} != {token_ids.numel()}"
            )

        width = values.numel()
        if params.top_k is not None and params.top_k > 0:
            width = min(width, params.top_k)
        values = self._sanitize_logits(values[:width])
        token_ids = token_ids[:width]
        if params.temperature <= 0.0:
            return int(token_ids[self._greedy_token(values)].item())

        scaled = values / max(params.temperature, 1e-5)
        probs = torch.softmax(scaled, dim=-1)
        if not self._is_valid_distribution(probs):
            return int(token_ids[self._greedy_token(values)].item())

        if 0.0 < params.top_p < 1.0:
            sorted_probs, sorted_positions = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= params.top_p
            keep[0] = True
            filtered_probs = torch.zeros_like(probs)
            filtered_probs[sorted_positions[keep]] = probs[sorted_positions[keep]]
            total = filtered_probs.sum()
            if not torch.isfinite(total) or total.item() <= 0.0:
                return int(token_ids[self._greedy_token(values)].item())
            probs = filtered_probs / total
            if not self._is_valid_distribution(probs):
                return int(token_ids[self._greedy_token(values)].item())

        sampled_pos = torch.multinomial(
            probs,
            num_samples=1,
            generator=self._generator(probs, params),
        )
        return int(token_ids[int(sampled_pos.item())].item())

    @staticmethod
    def _generator(values: torch.Tensor, params: SamplingParams) -> torch.Generator | None:
        """Build the deterministic request generator when a seed was supplied."""
        if params.seed is None:
            return None
        generator = torch.Generator(device=values.device)
        generator.manual_seed(int(params.seed) & 0x3FFFFFFF)
        return generator

    @staticmethod
    def from_generate_config(config: GenerateConfig) -> SamplingParams:
        """Build sampler parameters from user-facing generation config."""
        return SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            seed=config.seed,
        )

    @staticmethod
    def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
        """Replace non-finite logits with finite fallback values."""
        logits = logits.float()
        finite_mask = torch.isfinite(logits)
        if finite_mask.all():
            return logits
        warnings.warn("Sampler received non-finite logits; falling back to sanitized values.", stacklevel=2)
        if not finite_mask.any():
            return torch.zeros_like(logits)
        finite_logits = logits[finite_mask]
        floor = finite_logits.min().item() - 1e4
        return torch.nan_to_num(logits, nan=floor, neginf=floor, posinf=finite_logits.max().item())

    @staticmethod
    def _is_valid_distribution(probs: torch.Tensor) -> bool:
        """Return whether probabilities can be sampled safely."""
        total = probs.sum()
        return bool(torch.isfinite(probs).all() and torch.all(probs >= 0) and torch.isfinite(total) and total.item() > 0.0)

    @staticmethod
    def _greedy_token(logits: torch.Tensor) -> int:
        """Return the highest-logit token ID."""
        return int(torch.argmax(logits).item())
