# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math

import torch

try:
    from python.core.executor import ModelExecutor
    from python.core.kv_cache import KvCacheManager
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
    from python.core.types import (
        DecodeBatch,
        DecodeResult,
        LayerWeights,
        PrefillBatch,
        PrefillResult,
        RuntimeModel,
    )


class CpuModelExecutor(ModelExecutor):
    """Reference CPU executor for functional generation and small tests."""

    def __init__(self, kv_cache_manager: KvCacheManager, *, num_layers_override: int | None = None) -> None:
        super().__init__(kv_cache_manager)
        self._num_layers_override = num_layers_override

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
                if self._num_layers_override is not None and layer_idx >= self._num_layers_override:
                    break
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
            logits_rows.append(self._project_logits(model, last_hidden))
        return PrefillResult(last_hidden=torch.stack(last_hidden_rows), logits=torch.stack(logits_rows))

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run one autoregressive decode step for each active request."""
        hidden_rows: list[torch.Tensor] = []
        logits_rows: list[torch.Tensor] = []
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            hidden = batch.hidden_states[batch_idx].to(model.runtime.device).float()
            position = int(batch.seq_lens[batch_idx].item()) - 1

            for layer_idx, layer in enumerate(model.layers):
                if self._num_layers_override is not None and layer_idx >= self._num_layers_override:
                    break
                hidden = self._layer_decode(
                    model=model,
                    layer_idx=layer_idx,
                    layer=layer,
                    hidden_state=hidden,
                    position=position,
                    alloc=alloc,
                )

            hidden_rows.append(hidden)
            logits_rows.append(self._project_logits(model, hidden))
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
        normed = self._rms_norm(hidden_states, layer.input_rms_weight, config.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(-1, config.num_attention_heads, config.head_dim)
        k = self._linear(normed, layer.wk).view(-1, config.num_key_value_heads, config.head_dim)
        v = self._linear(normed, layer.wv).view(-1, config.num_key_value_heads, config.head_dim)
        q = self._per_head_rms_norm(q, layer.q_norm_weight, config.rms_norm_eps)
        k = self._per_head_rms_norm(k, layer.k_norm_weight, config.rms_norm_eps)
        q = self._apply_rope(q, positions, config.rope_theta)
        k = self._apply_rope(k, positions, config.rope_theta)
        self._kv_cache_manager.write_tokens(layer_idx, alloc, 0, k.to(model.runtime.device), v.to(model.runtime.device))
        attn_out = self._attention_prefill(q, k, v, config.num_attention_heads, config.num_key_value_heads)
        attn_resid = hidden_states + self._linear(attn_out.reshape(hidden_states.shape[0], -1), layer.wo)
        mlp_normed = self._rms_norm(attn_resid, layer.post_rms_weight, config.rms_norm_eps)
        gate = self._linear(mlp_normed, layer.w_gate)
        up = self._linear(mlp_normed, layer.w_up)
        mlp = torch.nn.functional.silu(gate) * up
        return attn_resid + self._linear(mlp, layer.w_down)

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
        normed = self._rms_norm(hidden_state.unsqueeze(0), layer.input_rms_weight, config.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(config.num_attention_heads, config.head_dim)
        k = self._linear(normed, layer.wk).view(config.num_key_value_heads, config.head_dim)
        v = self._linear(normed, layer.wv).view(config.num_key_value_heads, config.head_dim)
        q = self._per_head_rms_norm(q.unsqueeze(0), layer.q_norm_weight, config.rms_norm_eps).squeeze(0)
        k = self._per_head_rms_norm(k.unsqueeze(0), layer.k_norm_weight, config.rms_norm_eps).squeeze(0)
        pos = torch.tensor([position], device=model.runtime.device, dtype=torch.long)
        q = self._apply_rope(q.unsqueeze(0), pos, config.rope_theta).squeeze(0)
        k = self._apply_rope(k.unsqueeze(0), pos, config.rope_theta).squeeze(0)
        self._kv_cache_manager.write_tokens(
            layer_idx,
            alloc,
            position,
            k.unsqueeze(0).to(model.runtime.device),
            v.unsqueeze(0).to(model.runtime.device),
        )
        k_ctx, v_ctx = self._kv_cache_manager.read_context(layer_idx, alloc)
        attn_out = self._attention_decode(q, k_ctx, v_ctx, config.num_attention_heads, config.num_key_value_heads)
        attn_resid = hidden_state + self._linear(attn_out.reshape(1, -1), layer.wo).squeeze(0)
        mlp_normed = self._rms_norm(attn_resid.unsqueeze(0), layer.post_rms_weight, config.rms_norm_eps)
        gate = self._linear(mlp_normed, layer.w_gate)
        up = self._linear(mlp_normed, layer.w_up)
        mlp = torch.nn.functional.silu(gate) * up
        return attn_resid + self._linear(mlp, layer.w_down).squeeze(0)

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
# TurboQuant variant
# ---------------------------------------------------------------------------

try:
    from python.core.turboquant.compressor import generate_rotation_matrix
    from python.core.turboquant.lloyd_max import LloydMaxCodebook
except ImportError:
    from python.core.turboquant.compressor import generate_rotation_matrix
    from python.core.turboquant.lloyd_max import LloydMaxCodebook


HEAD_DIM_TQ = 128
EPS_TQ = 1e-6


class CpuTqModelExecutor(CpuModelExecutor):
    """CPU executor with TurboQuant KV cache compression (no dumps)."""

    def __init__(self, kv_cache_manager, *, num_layers_override=None):
        super().__init__(kv_cache_manager, num_layers_override=num_layers_override)
        self._codebook = None
        self._rot_matrices = []

    def _ensure_codebook(self, head_dim):
        if self._codebook is None:
            self._codebook = LloydMaxCodebook(head_dim, bits=4)

    def _ensure_rot_matrices(self, num_layers, head_dim):
        if len(self._rot_matrices) == num_layers:
            return
        self._rot_matrices = [
            generate_rotation_matrix(head_dim, seed=42 + l * 1000).bfloat16()
            for l in range(num_layers)
        ]

    def _tq_quantize(self, x, rot_matrix):
        cb = self._codebook.centroids
        bnd = self._codebook.boundaries
        l2 = torch.sqrt(x.float().pow(2).sum(dim=-1, keepdim=True) + EPS_TQ)
        xr = torch.matmul((x.float() / l2).bfloat16(), rot_matrix).float()
        idx = torch.searchsorted(bnd, xr).to(torch.int32).to(torch.float16).to(torch.uint8)
        return idx, l2.float()

    def _tq_dequantize(self, qi, sc, rm):
        cb = self._codebook.centroids
        yh = cb[qi.to(torch.float16).to(torch.int32).long()]
        yh = yh.float() / torch.sqrt(yh.float().pow(2).sum(dim=-1, keepdim=True) + EPS_TQ)
        return torch.matmul(yh, rm.float().T) * sc

    def run_prefill(self, model, batch):
        cfg = model.config
        self._ensure_codebook(cfg.head_dim)
        self._ensure_rot_matrices(cfg.num_hidden_layers, cfg.head_dim)
        last_hidden_rows, logits_rows = [], []
        for bi, alloc in enumerate(batch.kv_allocations):
            h = batch.input_embeddings[bi].to(model.runtime.device).float()
            seq_len = int(batch.seq_lens[bi].item())
            h = h[:seq_len]
            pos = torch.arange(seq_len, device=model.runtime.device, dtype=torch.long)
            for li, layer in enumerate(model.layers):
                if self._num_layers_override is not None and li >= self._num_layers_override:
                    break
                h = self._layer_prefill_tq(model, li, layer, h, pos, alloc)
            last_hidden_rows.append(h[-1])
            logits_rows.append(self._project_logits(model, h[-1]))
        return PrefillResult(last_hidden=torch.stack(last_hidden_rows), logits=torch.stack(logits_rows))

    def run_decode(self, model, batch):
        hidden_rows, logits_rows = [], []
        for bi, alloc in enumerate(batch.kv_allocations):
            h = batch.hidden_states[bi].to(model.runtime.device).float()
            pos = int(batch.seq_lens[bi].item()) - 1
            for li, layer in enumerate(model.layers):
                if self._num_layers_override is not None and li >= self._num_layers_override:
                    break
                h = self._layer_decode_tq(model, li, layer, h, pos, alloc)
            hidden_rows.append(h)
            logits_rows.append(self._project_logits(model, h))
        return DecodeResult(hidden_states=torch.stack(hidden_rows), logits=torch.stack(logits_rows))

    def _layer_prefill_tq(self, model, layer_idx, layer, hidden_states, positions, alloc):
        cfg = model.config
        nkh, nh, hd = cfg.num_key_value_heads, cfg.num_attention_heads, cfg.head_dim
        rm = self._rot_matrices[layer_idx]
        sl = hidden_states.shape[0]
        normed = self._rms_norm(hidden_states, layer.input_rms_weight, cfg.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(-1, nh, hd)
        k = self._linear(normed, layer.wk).view(-1, nkh, hd)
        v = self._linear(normed, layer.wv).view(-1, nkh, hd)
        q = self._per_head_rms_norm(q, layer.q_norm_weight, cfg.rms_norm_eps)
        k = self._per_head_rms_norm(k, layer.k_norm_weight, cfg.rms_norm_eps)
        q = self._apply_rope(q, positions, cfg.rope_theta)
        k = self._apply_rope(k, positions, cfg.rope_theta)
        qk_all = torch.zeros(sl, nkh, hd, dtype=torch.uint8)
        qv_all = torch.zeros(sl, nkh, hd, dtype=torch.uint8)
        ks_all = torch.zeros(sl, nkh, 1)
        vs_all = torch.zeros(sl, nkh, 1)
        for hx in range(nkh):
            qi, ks = self._tq_quantize(k[:, hx, :].reshape(-1, hd), rm)
            qk_all[:, hx, :] = qi; ks_all[:, hx, :] = ks
            qi, vs = self._tq_quantize(v[:, hx, :].reshape(-1, hd), rm)
            qv_all[:, hx, :] = qi; vs_all[:, hx, :] = vs
        pool = self._kv_cache_manager._pool(model.config.model_id)
        if pool.quant_key_indices is not None:
            self._write_quant_cache(pool, layer_idx, alloc, 0, qk_all, qv_all, ks_all, vs_all)
        ao = self._attention_prefill_tq(q, qk_all, qv_all, ks_all, vs_all, nh, nkh, rm)
        ar = hidden_states + self._linear(ao.reshape(sl, -1), layer.wo)
        mn = self._rms_norm(ar, layer.post_rms_weight, cfg.rms_norm_eps)
        g, u = self._linear(mn, layer.w_gate), self._linear(mn, layer.w_up)
        return ar + self._linear(torch.nn.functional.silu(g) * u, layer.w_down)

    def _layer_decode_tq(self, model, layer_idx, layer, hidden_state, position, alloc):
        cfg = model.config
        nkh, nh, hd = cfg.num_key_value_heads, cfg.num_attention_heads, cfg.head_dim
        rm = self._rot_matrices[layer_idx]
        normed = self._rms_norm(hidden_state.unsqueeze(0), layer.input_rms_weight, cfg.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(nh, hd)
        k = self._linear(normed, layer.wk).view(nkh, hd)
        v = self._linear(normed, layer.wv).view(nkh, hd)
        q = self._per_head_rms_norm(q.unsqueeze(0), layer.q_norm_weight, cfg.rms_norm_eps).squeeze(0)
        k = self._per_head_rms_norm(k.unsqueeze(0), layer.k_norm_weight, cfg.rms_norm_eps).squeeze(0)
        pt = torch.tensor([position], dtype=torch.long)
        q = self._apply_rope(q.unsqueeze(0), pt, cfg.rope_theta).squeeze(0)
        k = self._apply_rope(k.unsqueeze(0), pt, cfg.rope_theta).squeeze(0)
        nqk, nqv = torch.zeros(nkh, hd, dtype=torch.uint8), torch.zeros(nkh, hd, dtype=torch.uint8)
        nks, nvs = torch.zeros(nkh, 1), torch.zeros(nkh, 1)
        for hx in range(nkh):
            qi, ks = self._tq_quantize(k[hx:hx+1], rm); nqk[hx]=qi[0]; nks[hx]=ks[0]
            qi, vs = self._tq_quantize(v[hx:hx+1], rm); nqv[hx]=qi[0]; nvs[hx]=vs[0]
        pool = self._kv_cache_manager._pool(model.config.model_id)
        if pool.quant_key_indices is not None:
            self._write_quant_cache_single(pool, layer_idx, alloc, position, nqk, nqv, nks, nvs)
        qkf, qvf, ksf, vsf = self._read_full_quant_cache(pool, layer_idx, alloc, nkh, hd)
        ao = self._attention_decode_tq(q, qkf, qvf, ksf, vsf, nh, nkh, rm)
        ar = hidden_state + self._linear(ao.reshape(1, -1), layer.wo).squeeze(0)
        mn = self._rms_norm(ar.unsqueeze(0), layer.post_rms_weight, cfg.rms_norm_eps)
        g, u = self._linear(mn, layer.w_gate), self._linear(mn, layer.w_up)
        return ar + self._linear(torch.nn.functional.silu(g) * u, layer.w_down).squeeze(0)

    def _attention_prefill_tq(self, q, qk, qv, ks, vs, nh, nkh, rm):
        sl, qpk = q.shape[0], nh // nkh
        ctx = torch.zeros(sl, nh, HEAD_DIM_TQ, dtype=torch.float32)
        for kvh in range(nkh):
            kd = self._tq_dequantize(qk[:, kvh, :], ks[:, kvh, :], rm)
            vd = self._tq_dequantize(qv[:, kvh, :], vs[:, kvh, :], rm)
            qs = kvh * qpk
            qh = q[:, qs:qs + qpk, :]
            for i in range(sl):
                sc = torch.matmul(qh[i], kd[:i+1].T) / (HEAD_DIM_TQ ** 0.5)
                ctx[i, qs:qs+qpk, :] = torch.matmul(torch.softmax(sc, dim=-1), vd[:i+1])
        return ctx

    def _attention_decode_tq(self, q, qk, qv, ks, vs, nh, nkh, rm):
        qpk = nh // nkh
        parts = []
        for kvh in range(nkh):
            kd = self._tq_dequantize(qk[:, kvh, :], ks[:, kvh, :], rm)
            vd = self._tq_dequantize(qv[:, kvh, :], vs[:, kvh, :], rm)
            qh = q[kvh * qpk:(kvh + 1) * qpk]
            sc = torch.matmul(qh, kd.T) / (HEAD_DIM_TQ ** 0.5)
            parts.append(torch.matmul(torch.softmax(sc, dim=-1), vd))
        return torch.cat(parts, dim=0)

    @staticmethod
    def _write_quant_cache(pool, layer_idx, alloc, start_token, qk, qv, ks, vs):
        sl, nkh, hd = qk.shape
        ps = pool.page_size
        for to in range(sl):
            ti = start_token + to
            pg = ti // ps; off = ti % ps
            if pg >= len(alloc.page_ids): break
            pp = alloc.page_ids[pg]
            for hx in range(nkh):
                pool.quant_key_indices[layer_idx, pp, hx, off, :] = qk[to, hx, :]
                pool.quant_val_indices[layer_idx, pp, hx, off, :] = qv[to, hx, :]
                pool.quant_key_norms[layer_idx, pp, hx, off, 0] = ks[to, hx, 0].to(torch.bfloat16)
                pool.quant_val_norms[layer_idx, pp, hx, off, 0] = vs[to, hx, 0].to(torch.bfloat16)
        alloc.tokens_used = max(alloc.tokens_used, start_token + sl)

    @staticmethod
    def _write_quant_cache_single(pool, layer_idx, alloc, pos, qk, qv, ks, vs):
        nkh, hd = qk.shape
        ps = pool.page_size
        pg, off = pos // ps, pos % ps
        if pg >= len(alloc.page_ids): return
        pp = alloc.page_ids[pg]
        for hx in range(nkh):
            pool.quant_key_indices[layer_idx, pp, hx, off, :] = qk[hx]
            pool.quant_val_indices[layer_idx, pp, hx, off, :] = qv[hx]
            pool.quant_key_norms[layer_idx, pp, hx, off, 0] = ks[hx, 0].to(torch.bfloat16)
            pool.quant_val_norms[layer_idx, pp, hx, off, 0] = vs[hx, 0].to(torch.bfloat16)
        alloc.tokens_used = max(alloc.tokens_used, pos + 1)

    @staticmethod
    def _read_full_quant_cache(pool, layer_idx, alloc, nkh, hd):
        tu, ps = alloc.tokens_used, pool.page_size
        qk = torch.zeros(tu, nkh, hd, dtype=torch.uint8)
        qv = torch.zeros(tu, nkh, hd, dtype=torch.uint8)
        ks = torch.zeros(tu, nkh, 1)
        vs = torch.zeros(tu, nkh, 1)
        for tok in range(tu):
            pg, off = tok // ps, tok % ps
            if pg >= len(alloc.page_ids): break
            pp = alloc.page_ids[pg]
            for hx in range(nkh):
                qk[tok, hx, :] = pool.quant_key_indices[layer_idx, pp, hx, off, :]
                qv[tok, hx, :] = pool.quant_val_indices[layer_idx, pp, hx, off, :]
                ks[tok, hx, 0] = pool.quant_key_norms[layer_idx, pp, hx, off, 0].float()
                vs[tok, hx, 0] = pool.quant_val_norms[layer_idx, pp, hx, off, 0].float()
        return qk, qv, ks, vs
