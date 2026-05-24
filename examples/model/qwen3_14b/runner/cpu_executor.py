# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from pathlib import Path

import torch

try:
    from python.core.executor import ModelExecutor
    from python.core.kv_cache import KvCacheManager
    from python.core.turboquant.compressor import generate_rotation_matrix
    from python.core.turboquant.lloyd_max import LloydMaxCodebook
    from python.core.types import (
        DecodeBatch,
        DecodeResult,
        LayerWeights,
        PrefillBatch,
        PrefillResult,
        RuntimeModel,
    )
except ImportError:
    from python.core.executor import ModelExecutor
    from python.core.kv_cache import KvCacheManager
    from python.core.turboquant.compressor import generate_rotation_matrix
    from python.core.turboquant.lloyd_max import LloydMaxCodebook
    from python.core.types import (
        DecodeBatch,
        DecodeResult,
        LayerWeights,
        PrefillBatch,
        PrefillResult,
        RuntimeModel,
    )


HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
EPS = 1e-6


class CpuModelExecutor(ModelExecutor):
    """Reference CPU executor for functional generation and small tests."""

    def __init__(self, kv_cache_manager: KvCacheManager, *, dump_dir: str | None = None) -> None:
        super().__init__(kv_cache_manager)
        self._dump_dir = Path(dump_dir) if dump_dir else None
        if self._dump_dir is not None:
            self._dump_dir.mkdir(parents=True, exist_ok=True)

    def _dump_layer(self, layer_idx: int, step: str, phase: str, **tensors: torch.Tensor) -> None:
        """Save per-layer intermediate tensors to disk if dump_dir is set.

        Args:
            layer_idx: Transformer layer index.
            step: 'prefill' or 'decode'.
            phase: Sub-step within the layer (e.g. 'qkv', 'attn', 'mlp').
            **tensors: Named tensors to save.
        """
        if self._dump_dir is None:
            return
        path = self._dump_dir / f"layer{layer_idx:02d}_{step}_{phase}.pt"
        torch.save({k: v.detach().cpu().clone() for k, v in tensors.items()}, path)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run each prompt through all transformer layers on CPU."""
        last_hidden_rows: list[torch.Tensor] = []
        logits_rows: list[torch.Tensor] = []
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            hidden = batch.input_embeddings[batch_idx].to(model.runtime.device).float()
            seq_len = int(batch.seq_lens[batch_idx].item())
            hidden = hidden[:seq_len]
            positions = torch.arange(seq_len, device=model.runtime.device, dtype=torch.long)

            for layer_idx, layer in enumerate(model.layers):
                hidden = self._layer_prefill(
                    model=model,
                    layer_idx=layer_idx,
                    layer=layer,
                    hidden_states=hidden,
                    positions=positions,
                    alloc=alloc,
                )

            last_hidden = hidden[-1]
            last_hidden_rows.append(last_hidden)
            logits = self._project_logits(model, last_hidden)
            if self._dump_dir is not None:
                torch.save({"logits": logits.detach().cpu().clone()}, self._dump_dir / "logits_prefill.pt")
            logits_rows.append(logits)
        return PrefillResult(last_hidden=torch.stack(last_hidden_rows), logits=torch.stack(logits_rows))

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run one autoregressive decode step for each active request."""
        hidden_rows: list[torch.Tensor] = []
        logits_rows: list[torch.Tensor] = []
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            hidden = batch.hidden_states[batch_idx].to(model.runtime.device).float()
            position = int(batch.seq_lens[batch_idx].item()) - 1

            for layer_idx, layer in enumerate(model.layers):
                hidden = self._layer_decode(
                    model=model,
                    layer_idx=layer_idx,
                    layer=layer,
                    hidden_state=hidden,
                    position=position,
                    alloc=alloc,
                )

            hidden_rows.append(hidden)
            logits = self._project_logits(model, hidden)
            if self._dump_dir is not None and len(hidden_rows) == 1:
                torch.save({"logits": logits.detach().cpu().clone()}, self._dump_dir / f"logits_decode{position}.pt")
            logits_rows.append(logits)
        return DecodeResult(hidden_states=torch.stack(hidden_rows), logits=torch.stack(logits_rows))

    def _layer_prefill(
        self,
        model: RuntimeModel,
        layer_idx: int,
        layer: LayerWeights,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        alloc,
    ) -> torch.Tensor:
        """Run one transformer layer over a full prompt and write KV cache."""
        config = model.config
        self._dump_layer(layer_idx, "prefill", "input", hidden_states=hidden_states)
        normed = self._rms_norm(hidden_states, layer.input_rms_weight, config.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(-1, config.num_attention_heads, config.head_dim)
        k = self._linear(normed, layer.wk).view(-1, config.num_key_value_heads, config.head_dim)
        v = self._linear(normed, layer.wv).view(-1, config.num_key_value_heads, config.head_dim)
        self._dump_layer(layer_idx, "prefill", "qkv_raw", q=q, k=k, v=v)
        q = self._per_head_rms_norm(q, layer.q_norm_weight, config.rms_norm_eps)
        k = self._per_head_rms_norm(k, layer.k_norm_weight, config.rms_norm_eps)
        self._dump_layer(layer_idx, "prefill", "qkv_normed", q=q, k=k)
        q = self._apply_rope(q, positions, config.rope_theta)
        k = self._apply_rope(k, positions, config.rope_theta)
        self._dump_layer(layer_idx, "prefill", "qkv_rope", q=q, k=k, v=v)
        self._kv_cache_manager.write_tokens(layer_idx, alloc, 0, k.to(model.runtime.device), v.to(model.runtime.device))
        attn_out = self._attention_prefill(q, k, v, config.num_attention_heads, config.num_key_value_heads)
        self._dump_layer(layer_idx, "prefill", "attn_out", attn_out=attn_out)
        attn_resid = hidden_states + self._linear(attn_out.reshape(hidden_states.shape[0], -1), layer.wo)
        self._dump_layer(layer_idx, "prefill", "attn_resid", attn_resid=attn_resid)
        mlp_normed = self._rms_norm(attn_resid, layer.post_rms_weight, config.rms_norm_eps)
        gate = self._linear(mlp_normed, layer.w_gate)
        up = self._linear(mlp_normed, layer.w_up)
        mlp = torch.nn.functional.silu(gate) * up
        out = attn_resid + self._linear(mlp, layer.w_down)
        self._dump_layer(layer_idx, "prefill", "output", output=out)
        return out

    def _layer_decode(
        self,
        model: RuntimeModel,
        layer_idx: int,
        layer: LayerWeights,
        hidden_state: torch.Tensor,
        position: int,
        alloc,
    ) -> torch.Tensor:
        """Run one transformer layer for a single decode position."""
        config = model.config
        self._dump_layer(layer_idx, f"decode{position}", "input", hidden_state=hidden_state)
        normed = self._rms_norm(hidden_state.unsqueeze(0), layer.input_rms_weight, config.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(config.num_attention_heads, config.head_dim)
        k = self._linear(normed, layer.wk).view(config.num_key_value_heads, config.head_dim)
        v = self._linear(normed, layer.wv).view(config.num_key_value_heads, config.head_dim)
        self._dump_layer(layer_idx, f"decode{position}", "qkv_raw", q=q, k=k, v=v)
        q = self._per_head_rms_norm(q.unsqueeze(0), layer.q_norm_weight, config.rms_norm_eps).squeeze(0)
        k = self._per_head_rms_norm(k.unsqueeze(0), layer.k_norm_weight, config.rms_norm_eps).squeeze(0)
        self._dump_layer(layer_idx, f"decode{position}", "qkv_normed", q=q, k=k)
        pos = torch.tensor([position], device=model.runtime.device, dtype=torch.long)
        q = self._apply_rope(q.unsqueeze(0), pos, config.rope_theta).squeeze(0)
        k = self._apply_rope(k.unsqueeze(0), pos, config.rope_theta).squeeze(0)
        self._dump_layer(layer_idx, f"decode{position}", "qkv_rope", q=q, k=k, v=v)
        self._kv_cache_manager.write_tokens(
            layer_idx,
            alloc,
            position,
            k.unsqueeze(0).to(model.runtime.device),
            v.unsqueeze(0).to(model.runtime.device),
        )
        k_ctx, v_ctx = self._kv_cache_manager.read_context(layer_idx, alloc)
        attn_out = self._attention_decode(q, k_ctx, v_ctx, config.num_attention_heads, config.num_key_value_heads)
        self._dump_layer(layer_idx, f"decode{position}", "attn_out", attn_out=attn_out)
        attn_resid = hidden_state + self._linear(attn_out.reshape(1, -1), layer.wo).squeeze(0)
        self._dump_layer(layer_idx, f"decode{position}", "attn_resid", attn_resid=attn_resid)
        mlp_normed = self._rms_norm(attn_resid.unsqueeze(0), layer.post_rms_weight, config.rms_norm_eps)
        gate = self._linear(mlp_normed, layer.w_gate)
        up = self._linear(mlp_normed, layer.w_up)
        mlp = torch.nn.functional.silu(gate) * up
        out = attn_resid + self._linear(mlp, layer.w_down).squeeze(0)
        self._dump_layer(layer_idx, f"decode{position}", "output", output=out)
        return out

    def _project_logits(self, model: RuntimeModel, hidden: torch.Tensor) -> torch.Tensor:
        """Apply final RMS norm and LM head projection."""
        squeeze = hidden.dim() == 1
        hidden_2d = hidden.unsqueeze(0) if squeeze else hidden
        normed = self._rms_norm(hidden_2d, model.final_norm_weight, model.config.rms_norm_eps)
        logits = self._linear(normed, model.lm_head)
        return logits.squeeze(0) if squeeze else logits

    @staticmethod
    def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply a dense projection using Hugging Face weight orientation."""
        return x.float() @ weight.float().transpose(0, 1)

    @staticmethod
    def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply RMSNorm over the hidden dimension."""
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(variance + eps) * weight.float().view(1, -1)

    @staticmethod
    def _per_head_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply RMSNorm independently to each attention head."""
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(variance + eps) * weight.float().view(1, 1, -1)

    @staticmethod
    def _apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
        """Apply rotary position embedding to query or key heads."""
        head_dim = x.shape[-1]
        half = head_dim // 2
        device = x.device
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        freqs = torch.outer(positions.float(), inv_freq)
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        x_lo = x[..., :half]
        x_hi = x[..., half:]
        return torch.cat([x_lo * cos - x_hi * sin, x_hi * cos + x_lo * sin], dim=-1)

    @staticmethod
    def _attention_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        """Compute causal full-sequence attention for prompt prefill."""
        q_per_kv = num_heads // num_kv_heads
        k_rep = k.repeat_interleave(q_per_kv, dim=1).permute(1, 0, 2)
        v_rep = v.repeat_interleave(q_per_kv, dim=1).permute(1, 0, 2)
        q_heads = q.permute(1, 0, 2)
        scores = torch.matmul(q_heads, k_rep.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        seq_len = q.shape[0]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v_rep).permute(1, 0, 2)

    @staticmethod
    def _attention_decode(
        q: torch.Tensor,
        k_ctx: torch.Tensor,
        v_ctx: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        """Compute attention for one query against cached context."""
        k_ctx = k_ctx.float()
        v_ctx = v_ctx.float()
        q_per_kv = num_heads // num_kv_heads
        k_rep = k_ctx.repeat_interleave(q_per_kv, dim=1).permute(1, 0, 2)
        v_rep = v_ctx.repeat_interleave(q_per_kv, dim=1).permute(1, 0, 2)
        q_heads = q.unsqueeze(1)
        scores = torch.matmul(q_heads, k_rep.transpose(-1, -2)).squeeze(1) / math.sqrt(q.shape[-1])
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn.unsqueeze(1), v_rep).squeeze(1)


# ---------------------------------------------------------------------------
# TurboQuant variant — inherits the FP helpers above and overrides the layer
# forward to quantize/dequantize K/V. Kept in this file to avoid duplicating
# _dump_layer / _project_logits / _linear / _rms_norm / _apply_rope.
# ---------------------------------------------------------------------------


class CpuTqModelExecutor(CpuModelExecutor):
    """CPU executor with TurboQuant KV cache compression.

    Extends CpuModelExecutor by compressing K/V into quantized format
    (UINT8 indices + FP32 L2 norms) after projection, and dequantizing
    during attention. Uses the same PolarQuant algorithm as the NPU kernel:
      L2 norm → normalize → rotate → Lloyd-Max quantize → store UINT8 + scale
      Attention: gather(codebook) × scale → dequant → QK/SV → unrotate context
    """

    def __init__(
        self,
        kv_cache_manager: KvCacheManager,
        *,
        dump_dir: str | None = None,
        num_layers_override: int | None = None,
    ) -> None:
        super().__init__(kv_cache_manager, dump_dir=dump_dir)
        self._codebook: LloydMaxCodebook | None = None
        self._rot_matrices: list[torch.Tensor] = []  # per-layer [HEAD_DIM, HEAD_DIM] BF16
        self._kv_quant_config = None
        self._num_layers_override = num_layers_override

    def _ensure_codebook(self, head_dim: int) -> None:
        """Lazily create the Lloyd-Max codebook."""
        if self._codebook is None:
            self._codebook = LloydMaxCodebook(head_dim, bits=4)

    def _ensure_rot_matrices(self, num_layers: int, head_dim: int, seed: int = 42) -> None:
        """Lazily create per-layer rotation matrices."""
        if len(self._rot_matrices) == num_layers:
            return
        self._rot_matrices = [
            generate_rotation_matrix(head_dim, seed=seed + l * 1000).bfloat16()
            for l in range(num_layers)
        ]

    # ------------------------------------------------------------------
    # Quantize / Dequantize primitives
    # ------------------------------------------------------------------

    def _tq_quantize(
        self,
        x: torch.Tensor,
        rot_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K or V vectors: L2 norm → normalize → rotate → Lloyd-Max.

        Args:
            x: [N, HEAD_DIM] FP32 (K after RoPE, or V without RoPE)
            rot_matrix: [HEAD_DIM, HEAD_DIM] BF16

        Returns:
            quant_indices: [N, HEAD_DIM] UINT8
            scales: [N, 1] FP32 (L2 norms)
        """
        centroids = self._codebook.centroids
        boundaries = self._codebook.boundaries

        # L2 norm
        l2_norm = torch.sqrt(x.float().pow(2).sum(dim=-1, keepdim=True) + EPS)
        x_norm = x.float() / l2_norm

        # Rotate (BF16 matmul like NPU)
        x_rot = torch.matmul(x_norm.bfloat16(), rot_matrix).float()

        # Lloyd-Max quantize
        indices = torch.searchsorted(boundaries, x_rot)
        indices_u8 = indices.to(torch.int32).to(torch.float16).to(torch.uint8)

        return indices_u8, l2_norm.float()

    def _tq_dequantize(
        self,
        quant_indices: torch.Tensor,
        scales: torch.Tensor,
        rot_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize matching the reference polar_quant.py pipeline.

        Reference pipeline:
          y_hat = centroids[indices]
          y_hat = y_hat / ||y_hat||            (renormalize to unit sphere)
          x_hat_unit = y_hat @ rot_matrix.T    (unrotate, right-multiply transpose)
          x_hat = x_hat_unit * norms           (rescale)

        Args:
            quant_indices: [N, HEAD_DIM] UINT8
            scales: [N, 1] FP32 (L2 norms)
            rot_matrix: [HEAD_DIM, HEAD_DIM] BF16 rotation matrix

        Returns:
            [N, HEAD_DIM] FP32 (in original space)
        """
        centroids = self._codebook.centroids
        idx_int32 = quant_indices.to(torch.float16).to(torch.int32)
        y_hat = centroids[idx_int32.long()]  # [N, HEAD_DIM] in rotated space

        # Renormalize to unit sphere (critical for accuracy)
        y_norms = torch.sqrt(y_hat.float().pow(2).sum(dim=-1, keepdim=True) + EPS)
        y_hat = y_hat.float() / y_norms

        # Unrotate: right-multiply by rot_matrix.T
        x_hat_unit = torch.matmul(y_hat, rot_matrix.float().T)

        # Rescale by stored L2 norms
        return x_hat_unit * scales

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run each prompt through all transformer layers on CPU with TQ."""
        config = model.config
        num_layers = config.num_hidden_layers
        head_dim = config.head_dim

        self._ensure_codebook(head_dim)
        self._ensure_rot_matrices(num_layers, head_dim)

        last_hidden_rows: list[torch.Tensor] = []
        logits_rows: list[torch.Tensor] = []

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            hidden = batch.input_embeddings[batch_idx].to(model.runtime.device).float()
            seq_len = int(batch.seq_lens[batch_idx].item())
            hidden = hidden[:seq_len]
            positions = torch.arange(seq_len, device=model.runtime.device, dtype=torch.long)

            for layer_idx, layer in enumerate(model.layers):
                if self._num_layers_override is not None and layer_idx >= self._num_layers_override:
                    break
                hidden = self._layer_prefill_tq(
                    model=model,
                    layer_idx=layer_idx,
                    layer=layer,
                    hidden_states=hidden,
                    positions=positions,
                    alloc=alloc,
                )

            last_hidden = hidden[-1]
            last_hidden_rows.append(last_hidden)
            logits = self._project_logits(model, last_hidden)
            if self._dump_dir is not None:
                torch.save({"logits": logits.detach().cpu().clone()}, self._dump_dir / "logits_prefill.pt")
            logits_rows.append(logits)

        return PrefillResult(
            last_hidden=torch.stack(last_hidden_rows),
            logits=torch.stack(logits_rows),
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run one autoregressive decode step for each active request with TQ."""
        hidden_rows: list[torch.Tensor] = []
        logits_rows: list[torch.Tensor] = []

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            hidden = batch.hidden_states[batch_idx].to(model.runtime.device).float()
            position = int(batch.seq_lens[batch_idx].item()) - 1

            for layer_idx, layer in enumerate(model.layers):
                if self._num_layers_override is not None and layer_idx >= self._num_layers_override:
                    break
                hidden = self._layer_decode_tq(
                    model=model,
                    layer_idx=layer_idx,
                    layer=layer,
                    hidden_state=hidden,
                    position=position,
                    alloc=alloc,
                )

            hidden_rows.append(hidden)
            logits = self._project_logits(model, hidden)
            if self._dump_dir is not None and len(hidden_rows) == 1:
                torch.save({"logits": logits.detach().cpu().clone()}, self._dump_dir / f"logits_decode{position}.pt")
            logits_rows.append(logits)

        return DecodeResult(
            hidden_states=torch.stack(hidden_rows),
            logits=torch.stack(logits_rows),
        )

    # ------------------------------------------------------------------
    # TQ layer forward
    # ------------------------------------------------------------------

    def _layer_prefill_tq(
        self,
        model: RuntimeModel,
        layer_idx: int,
        layer: LayerWeights,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        alloc,
    ) -> torch.Tensor:
        """One transformer layer for prefill with TQ K/V compression."""
        config = model.config
        num_kv_heads = config.num_key_value_heads
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        seq_len = hidden_states.shape[0]
        rot_matrix = self._rot_matrices[layer_idx]

        # RMSNorm + QKV projection
        self._dump_layer(layer_idx, "prefill", "input", hidden_states=hidden_states)
        normed = self._rms_norm(hidden_states, layer.input_rms_weight, config.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(-1, num_heads, head_dim)
        k = self._linear(normed, layer.wk).view(-1, num_kv_heads, head_dim)
        v = self._linear(normed, layer.wv).view(-1, num_kv_heads, head_dim)
        self._dump_layer(layer_idx, "prefill", "qkv_raw", q=q, k=k, v=v)

        q = self._per_head_rms_norm(q, layer.q_norm_weight, config.rms_norm_eps)
        k = self._per_head_rms_norm(k, layer.k_norm_weight, config.rms_norm_eps)
        self._dump_layer(layer_idx, "prefill", "qkv_normed", q=q, k=k)

        q = self._apply_rope(q, positions, config.rope_theta)
        k = self._apply_rope(k, positions, config.rope_theta)
        self._dump_layer(layer_idx, "prefill", "qkv_rope", q=q, k=k, v=v)

        # ── TQ: Quantize K/V ──
        quant_k_all = torch.zeros(seq_len, num_kv_heads, head_dim, dtype=torch.uint8)
        quant_v_all = torch.zeros(seq_len, num_kv_heads, head_dim, dtype=torch.uint8)
        k_scales_all = torch.zeros(seq_len, num_kv_heads, 1, dtype=torch.float32)
        v_scales_all = torch.zeros(seq_len, num_kv_heads, 1, dtype=torch.float32)

        for h in range(num_kv_heads):
            qk, ks = self._tq_quantize(k[:, h, :].reshape(-1, head_dim), rot_matrix)
            quant_k_all[:, h, :] = qk
            k_scales_all[:, h, :] = ks
            qv, vs = self._tq_quantize(v[:, h, :].reshape(-1, head_dim), rot_matrix)
            quant_v_all[:, h, :] = qv
            v_scales_all[:, h, :] = vs

        self._dump_layer(layer_idx, "prefill", "quant_kv",
                         quant_k=quant_k_all, quant_v=quant_v_all,
                         k_scales=k_scales_all, v_scales=v_scales_all)

        # Dequantize and dump for comparison
        k_dequant = self._dequantize_all_heads(quant_k_all, k_scales_all, num_kv_heads, rot_matrix)
        v_dequant = self._dequantize_all_heads(quant_v_all, v_scales_all, num_kv_heads, rot_matrix)
        self._dump_layer(layer_idx, "prefill", "kv_dequant", k_dequant=k_dequant, v_dequant=v_dequant)

        # Write quantized cache via KvCacheManager
        pool = self._kv_cache_manager._pool(model.config.model_id)
        if pool.quant_key_indices is not None:
            self._write_quant_cache(
                pool, layer_idx, alloc, 0,
                quant_k_all, quant_v_all, k_scales_all, v_scales_all,
            )

        # ── TQ: Attention with dequantized K/V ──
        attn_out = self._attention_prefill_tq(
            q, quant_k_all, quant_v_all, k_scales_all, v_scales_all,
            num_heads, num_kv_heads, rot_matrix,
        )
        self._dump_layer(layer_idx, "prefill", "attn_out", attn_out=attn_out)

        attn_resid = hidden_states + self._linear(
            attn_out.reshape(seq_len, -1), layer.wo,
        )
        self._dump_layer(layer_idx, "prefill", "attn_resid", attn_resid=attn_resid)
        mlp_normed = self._rms_norm(attn_resid, layer.post_rms_weight, config.rms_norm_eps)
        gate = self._linear(mlp_normed, layer.w_gate)
        up = self._linear(mlp_normed, layer.w_up)
        mlp = torch.nn.functional.silu(gate) * up
        out = attn_resid + self._linear(mlp, layer.w_down)
        self._dump_layer(layer_idx, "prefill", "output", output=out)
        return out

    def _layer_decode_tq(
        self,
        model: RuntimeModel,
        layer_idx: int,
        layer: LayerWeights,
        hidden_state: torch.Tensor,
        position: int,
        alloc,
    ) -> torch.Tensor:
        """One transformer layer for single-token decode with TQ."""
        config = model.config
        num_kv_heads = config.num_key_value_heads
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        rot_matrix = self._rot_matrices[layer_idx]

        self._dump_layer(layer_idx, f"decode{position}", "input", hidden_state=hidden_state)
        normed = self._rms_norm(hidden_state.unsqueeze(0), layer.input_rms_weight, config.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(num_heads, head_dim)
        k = self._linear(normed, layer.wk).view(num_kv_heads, head_dim)
        v = self._linear(normed, layer.wv).view(num_kv_heads, head_dim)
        self._dump_layer(layer_idx, f"decode{position}", "qkv_raw", q=q, k=k, v=v)

        q = self._per_head_rms_norm(q.unsqueeze(0), layer.q_norm_weight, config.rms_norm_eps).squeeze(0)
        k = self._per_head_rms_norm(k.unsqueeze(0), layer.k_norm_weight, config.rms_norm_eps).squeeze(0)
        self._dump_layer(layer_idx, f"decode{position}", "qkv_normed", q=q, k=k)

        pos = torch.tensor([position], dtype=torch.long)
        q = self._apply_rope(q.unsqueeze(0), pos, config.rope_theta).squeeze(0)
        k = self._apply_rope(k.unsqueeze(0), pos, config.rope_theta).squeeze(0)
        self._dump_layer(layer_idx, f"decode{position}", "qkv_rope", q=q, k=k, v=v)

        # ── TQ: Quantize new K/V token ──
        new_qk = torch.zeros(num_kv_heads, head_dim, dtype=torch.uint8)
        new_qv = torch.zeros(num_kv_heads, head_dim, dtype=torch.uint8)
        new_ks = torch.zeros(num_kv_heads, 1, dtype=torch.float32)
        new_vs = torch.zeros(num_kv_heads, 1, dtype=torch.float32)

        for h in range(num_kv_heads):
            qk, ks = self._tq_quantize(k[h:h + 1], rot_matrix)
            new_qk[h] = qk[0]
            new_ks[h] = ks[0]
            qv, vs = self._tq_quantize(v[h:h + 1], rot_matrix)
            new_qv[h] = qv[0]
            new_vs[h] = vs[0]

        self._dump_layer(layer_idx, f"decode{position}", "quant_kv",
                         quant_k=new_qk, quant_v=new_qv,
                         k_scales=new_ks, v_scales=new_vs)

        # Append to quant cache
        pool = self._kv_cache_manager._pool(model.config.model_id)
        if pool.quant_key_indices is not None:
            self._write_quant_cache_single(
                pool, layer_idx, alloc, position,
                new_qk, new_qv, new_ks, new_vs,
            )

        # ── TQ: Read full quant cache and dequantize for attention ──
        quant_k_full, quant_v_full, k_scales_full, v_scales_full = (
            self._read_full_quant_cache(pool, layer_idx, alloc, num_kv_heads, head_dim)
        )

        # Dequantize and dump for comparison with FP K/V
        k_dequant = self._dequantize_all_heads(quant_k_full, k_scales_full, num_kv_heads, rot_matrix)
        v_dequant = self._dequantize_all_heads(quant_v_full, v_scales_full, num_kv_heads, rot_matrix)
        self._dump_layer(layer_idx, f"decode{position}", "kv_dequant", k_dequant=k_dequant, v_dequant=v_dequant)

        # Attention
        attn_out = self._attention_decode_tq(
            q, quant_k_full, quant_v_full, k_scales_full, v_scales_full,
            num_heads, num_kv_heads, rot_matrix,
        )
        self._dump_layer(layer_idx, f"decode{position}", "attn_out", attn_out=attn_out)

        attn_resid = hidden_state + self._linear(attn_out.reshape(1, -1), layer.wo).squeeze(0)
        self._dump_layer(layer_idx, f"decode{position}", "attn_resid", attn_resid=attn_resid)
        mlp_normed = self._rms_norm(attn_resid.unsqueeze(0), layer.post_rms_weight, config.rms_norm_eps)
        gate = self._linear(mlp_normed, layer.w_gate)
        up = self._linear(mlp_normed, layer.w_up)
        mlp = torch.nn.functional.silu(gate) * up
        out = attn_resid + self._linear(mlp, layer.w_down).squeeze(0)
        self._dump_layer(layer_idx, f"decode{position}", "output", output=out)
        return out

    # ------------------------------------------------------------------
    # Helpers for dump
    # ------------------------------------------------------------------

    def _dequantize_all_heads(
        self,
        quant: torch.Tensor,
        scales: torch.Tensor,
        num_kv_heads: int,
        rot_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize all heads of K or V for dump comparison.

        Args:
            quant: [tokens, num_kv_heads, head_dim] UINT8
            scales: [tokens, num_kv_heads, 1] FP32
            num_kv_heads: number of KV heads
            rot_matrix: [HEAD_DIM, HEAD_DIM]

        Returns:
            [tokens, num_kv_heads, head_dim] FP32
        """
        parts = []
        for h in range(num_kv_heads):
            deq = self._tq_dequantize(quant[:, h, :], scales[:, h, :], rot_matrix)
            parts.append(deq.unsqueeze(1))
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # TQ attention
    # ------------------------------------------------------------------

    def _attention_prefill_tq(
        self,
        q: torch.Tensor,
        quant_k: torch.Tensor,
        quant_v: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        rot_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Causal attention for prefill using dequantized K/V.

        Args:
            q: [seq, num_heads, HEAD_DIM]
            quant_k: [seq, num_kv_heads, HEAD_DIM] UINT8
            quant_v: [seq, num_kv_heads, HEAD_DIM] UINT8
            k_scales: [seq, num_kv_heads, 1] FP32
            v_scales: [seq, num_kv_heads, 1] FP32
        """
        seq_len = q.shape[0]
        q_per_kv = num_heads // num_kv_heads

        ctx = torch.zeros(seq_len, num_heads, HEAD_DIM, dtype=torch.float32)

        for kvh in range(num_kv_heads):
            # Dequantize K and V for this head (back to original space)
            k_deq = self._tq_dequantize(quant_k[:, kvh, :], k_scales[:, kvh, :], rot_matrix)
            v_deq = self._tq_dequantize(quant_v[:, kvh, :], v_scales[:, kvh, :], rot_matrix)

            q_start = kvh * q_per_kv
            q_end = q_start + q_per_kv
            q_h = q[:, q_start:q_end, :]  # [seq, q_per_kv, HEAD_DIM]

            for i in range(seq_len):
                qi = q_h[i]                         # [q_per_kv, HEAD_DIM]
                ki = k_deq[:i + 1]                  # [i+1, HEAD_DIM]
                vi = v_deq[:i + 1]                  # [i+1, HEAD_DIM]
                scores = torch.matmul(qi, ki.T) / math.sqrt(HEAD_DIM)
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, vi)         # [q_per_kv, HEAD_DIM]
                ctx[i, q_start:q_end, :] = out

        # No unrotation needed: K/V are already in original space after dequant
        return ctx

    def _attention_decode_tq(
        self,
        q: torch.Tensor,
        quant_k: torch.Tensor,
        quant_v: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        rot_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Attention for one query token against full dequantized K/V cache.

        Args:
            q: [num_heads, HEAD_DIM]
            quant_k: [seq, num_kv_heads, HEAD_DIM] UINT8
            quant_v: [seq, num_kv_heads, HEAD_DIM] UINT8
        """
        q_per_kv = num_heads // num_kv_heads

        ctx_parts = []
        for kvh in range(num_kv_heads):
            k_deq = self._tq_dequantize(quant_k[:, kvh, :], k_scales[:, kvh, :], rot_matrix)
            v_deq = self._tq_dequantize(quant_v[:, kvh, :], v_scales[:, kvh, :], rot_matrix)

            q_start = kvh * q_per_kv
            q_h = q[q_start:q_start + q_per_kv]  # [q_per_kv, HEAD_DIM]

            scores = torch.matmul(q_h, k_deq.T) / math.sqrt(HEAD_DIM)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_deq)  # [q_per_kv, HEAD_DIM]
            ctx_parts.append(out)

        return torch.cat(ctx_parts, dim=0)  # [num_heads, HEAD_DIM]

    # ------------------------------------------------------------------
    # Quant cache read/write
    # ------------------------------------------------------------------

    @staticmethod
    def _write_quant_cache(
        pool,
        layer_idx: int,
        alloc,
        start_token: int,
        quant_k: torch.Tensor,
        quant_v: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
    ) -> None:
        """Write quantized K/V for multiple tokens into the paged quant cache."""
        seq_len = quant_k.shape[0]
        num_kv_heads = quant_k.shape[1]
        page_size = pool.page_size
        head_dim = quant_k.shape[2]

        for tok_offset in range(seq_len):
            token_index = start_token + tok_offset
            page_idx = token_index // page_size
            offset = token_index % page_size
            if page_idx >= len(alloc.page_ids):
                break
            physical_page = alloc.page_ids[page_idx]

            for h in range(num_kv_heads):
                # 5D indexing: [layers, blocks, kv_heads, page_size, head_dim]
                pool.quant_key_indices[layer_idx, physical_page, h, offset, :] = quant_k[tok_offset, h, :]
                pool.quant_val_indices[layer_idx, physical_page, h, offset, :] = quant_v[tok_offset, h, :]
                pool.quant_key_norms[layer_idx, physical_page, h, offset, 0] = k_scales[tok_offset, h, 0].to(torch.bfloat16)
                pool.quant_val_norms[layer_idx, physical_page, h, offset, 0] = v_scales[tok_offset, h, 0].to(torch.bfloat16)
        alloc.tokens_used = max(alloc.tokens_used, start_token + seq_len)

    @staticmethod
    def _write_quant_cache_single(
        pool,
        layer_idx: int,
        alloc,
        position: int,
        quant_k: torch.Tensor,
        quant_v: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
    ) -> None:
        """Write quantized K/V for a single token (decode)."""
        num_kv_heads = quant_k.shape[0]
        head_dim = quant_k.shape[1]
        page_size = pool.page_size

        page_idx = position // page_size
        offset = position % page_size
        if page_idx >= len(alloc.page_ids):
            return
        physical_page = alloc.page_ids[page_idx]

        for h in range(num_kv_heads):
            pool.quant_key_indices[layer_idx, physical_page, h, offset, :] = quant_k[h]
            pool.quant_val_indices[layer_idx, physical_page, h, offset, :] = quant_v[h]
            pool.quant_key_norms[layer_idx, physical_page, h, offset, 0] = k_scales[h, 0].to(torch.bfloat16)
            pool.quant_val_norms[layer_idx, physical_page, h, offset, 0] = v_scales[h, 0].to(torch.bfloat16)
        alloc.tokens_used = max(alloc.tokens_used, position + 1)

    @staticmethod
    def _read_full_quant_cache(
        pool,
        layer_idx: int,
        alloc,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read all quantized K/V for a request + layer.

        Returns:
            quant_k: [tokens_used, num_kv_heads, head_dim] UINT8
            quant_v: [tokens_used, num_kv_heads, head_dim] UINT8
            k_scales: [tokens_used, num_kv_heads, 1] FP32
            v_scales: [tokens_used, num_kv_heads, 1] FP32
        """
        tokens_used = alloc.tokens_used
        page_size = pool.page_size

        quant_k = torch.zeros(tokens_used, num_kv_heads, head_dim, dtype=torch.uint8)
        quant_v = torch.zeros(tokens_used, num_kv_heads, head_dim, dtype=torch.uint8)
        k_scales = torch.zeros(tokens_used, num_kv_heads, 1, dtype=torch.float32)
        v_scales = torch.zeros(tokens_used, num_kv_heads, 1, dtype=torch.float32)

        for tok in range(tokens_used):
            page_idx = tok // page_size
            offset = tok % page_size
            if page_idx >= len(alloc.page_ids):
                break
            physical_page = alloc.page_ids[page_idx]

            for h in range(num_kv_heads):
                quant_k[tok, h, :] = pool.quant_key_indices[layer_idx, physical_page, h, offset, :]
                quant_v[tok, h, :] = pool.quant_val_indices[layer_idx, physical_page, h, offset, :]
                k_scales[tok, h, 0] = pool.quant_key_norms[layer_idx, physical_page, h, offset, 0].float()
                v_scales[tok, h, 0] = pool.quant_val_norms[layer_idx, physical_page, h, offset, 0].float()

        return quant_k, quant_v, k_scales, v_scales
