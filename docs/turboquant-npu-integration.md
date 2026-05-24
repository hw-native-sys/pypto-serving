# TurboQuant NPU Kernel Integration — Design Document

## Summary

Add TurboQuant KV cache compression to the serving pipeline. When enabled, prefill writes compressed KV cache directly via `prefill_fwd_tq` on NPU; decode performs fused dequant attention via `qwen3_decode_tq`. All compression and decompression runs on NPU with no Python fallback.

## Motivation

KV cache is the primary memory bottleneck for long-context and high-concurrency LLM inference. TurboQuant reduces KV cache memory footprint by compressing KV vectors from BF16 (16-bit) to low-bit quantized format (4-bit/2-bit indices + FP16 norms), achieving 60-80% memory reduction.

This feature adds end-to-end TurboQuant support to the serving pipeline via NPU kernels:

1. **Memory efficiency**: TurboQuant compresses KV cache to ~3-4 bits per element (vs 16-bit BF16), enabling longer sequences or more concurrent requests within the same device memory budget.
2. **NPU-accelerated compression**: The `prefill_fwd_tq` kernel writes compressed KV cache directly during prefill — no intermediate BF16 storage or post-processing step. The `qwen3_decode_tq` kernel performs fused dequant attention over compressed + resident blocks in a single NPU call.
3. **No Python overhead**: All quantization and dequantization runs on NPU. There is no Python fallback path — the entire compress/decompress pipeline is handled by compiled kernels.

## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Before (BF16 only)                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Prefill ──► BF16 KV Cache (write)                      │
│                     │                                    │
│                     ▼                                    │
│  Decode ──► BF16 KV Cache (read/write)                  │
│                                                         │
│  Kernels: prefill_fwd + decode_fwd                      │
│  All KV data stored as BF16, no compression.            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    After (TurboQuant NPU)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Prefill ──► Quant KV Cache (UINT8 + BF16 norms)        │
│              (prefill_fwd_tq writes directly)            │
│                     │                                    │
│                     ▼                                    │
│  Decode ──► Hybrid Attention                            │
│             ├─ Compressed blocks: quant_k/v (UINT8)     │
│             └─ Resident blocks:   k/v_cache (BF16)      │
│                     │                                    │
│                     ▼                                    │
│            _project_logits(hidden → logits)              │
│                                                         │
│  Kernels: prefill_fwd_tq + qwen3_decode_tq              │
└─────────────────────────────────────────────────────────┘
```

### Kernel Signatures

#### `prefill_fwd_tq` (`pypto-lib/models/qwen3/14b/qwen3_14b_prefill_tq.py`)

- **Type**: `@pl.jit` multi-layer fused (same compilation path as `prefill_fwd`).
- **Function**: Runs all layers in one kernel call, writes KV to UINT8 quantized format, includes final_norm + lm_head.
- **Output**: logits (FP32) — same as regular prefill.

```
Parameter diff vs regular prefill_fwd:
  Removed: k_cache (BF16), v_cache (BF16)
  Added:   quant_k_cache (UINT8), quant_v_cache (UINT8)
           quant_k_scales (BF16), quant_v_scales (BF16)
           rot_matrices (BF16)
           tq_codebook (FP32), k_idx_scratch (INT32), v_idx_scratch (INT32)
```

#### `qwen3_decode_tq` (`pypto-lib/models/qwen3/14b/qwen3_14b_decode_tq.py`)

- **Type**: `@pl.program` multi-layer fused (internal loop over `decode_layer_tq`).
- **Function**: Hybrid attention — Phase 1 reads compressed blocks (fused dequant), Phase 2 reads BF16 resident blocks.
- **Output**: hidden (BF16) + kv_norms_out (BF16) — requires separate `_project_logits()`.

```
Parameter diff vs regular decode_fwd:
  Kept:   k_cache (BF16), v_cache (BF16) — resident blocks for new tokens
  Added:  quant_k_cache (UINT8), quant_v_cache (UINT8)
          quant_k_scales (BF16), quant_v_scales (BF16)
          cmp_block_table (INT32), cmp_num_blocks (INT32)
          rot_matrices (BF16)
  Removed: final_norm_weight, lm_head_weight (not fused in kernel)
  Changed: output is hidden + kv_norms_out instead of logits
```

### Data Flow

```
1. Initialization (register_model)
   ├── BF16 pages: full allocation (compiled kernel stride is fixed)
   ├── Quant buffers: UINT8 indices + BF16 norms
   ├── Rotation matrices: per-layer seed = 42 + layer_idx * 1000
   └── Scratch buffers: tq_codebook, k_idx_scratch, v_idx_scratch

2. Prefill (run_prefill)
   ├── use_tq = True → prefill_fwd_tq
   │   ├── Input:  hidden_states, weights, block_table, slot_mapping
   │   ├── Write:  quant_k_cache, quant_v_cache (UINT8)
   │   ├── Write:  quant_k_scales, quant_v_scales (BF16)
   │   └── Output: logits (FP32, includes final_norm + lm_head)
   └── Update quant_page_count (mark which pages are quantized)

3. Decode (run_decode)
   ├── use_tq = True → qwen3_decode_tq
   │   ├── Phase 1: read quant_k/v_cache → fused dequant → QK/SV matmul
   │   ├── Phase 2: read k/v_cache (BF16 resident) → standard attention
   │   ├── Write:  k/v_cache (BF16, new token KV)
   │   ├── Write:  kv_norms_out (norms for new token KV)
   │   └── Output: hidden (BF16)
   ├── _project_logits(hidden → logits)
   └── Return DecodeResult

4. Compressed Block Tracking
   ├── cmp_block_table: [batch * max_blocks_per_seq] INT32
   ├── cmp_num_blocks:  [batch] INT32
   └── Driven by _RequestQuantState.quant_page_count
```

### BF16 KV Cache Strategy

Prefill does not need BF16 KV cache (`prefill_fwd_tq` writes directly to quantized format).

Decode still requires BF16 `k_cache` / `v_cache`:
- Used for **resident blocks** (new decode tokens written to BF16).
- Periodically eligible for migration to quantized format (residual window, not yet implemented).

Full BF16 allocation is retained for now because `decode_tq` has the per-layer BF16 cache stride baked in at compile time.

## Changes

### Files Modified

| File | Change |
|------|--------|
| `npu_executor.py` | Kernel compilation: prefill_tq switched to `@pl.jit` path, decode_tq module name updated, removed standalone compress/decompress, added scratch buffers |
| `npu_runner.py` | Runtime dispatch: run_prefill/run_decode wired to TQ kernels, removed all Python fallbacks and run_tq_compress/run_tq_decompress |
| `compressor.py` | Simplified to config class: removed all compress/decompress Python implementations, kept only rotation matrix generation + bit-packing |
| `kv_cache.py` | Added get_compressed_block_info(), marked compress_to_quant() as deprecated |

### Code Removed (simplified during development)

- `run_tq_compress()` / `run_tq_decompress()` — standalone compress/decompress (no longer needed, kernels are fused)
- `TurboQuantCompressor.compress()` / `decompress()` — Python compress/decompress (replaced by NPU kernels)
- `TurboQuantCompressor.compress_npu()` / `decompress_npu()` — intermediate NPU wrappers (superseded by fused kernels)
- `KVCompressor.compress_layer()` / `decompress_layer()` — per-layer Python wrappers (no longer needed)
- `KVCompressor.compress_layer_npu()` / `decompress_layer_npu()` — per-layer NPU wrappers (superseded by fused kernels)
- `compress_to_quant()` implementation body — post-prefill Python quantization loop (no longer needed with direct-write prefill)

### Code Added (+311 lines)

- `_compile_prefill_tq_callable()` — new `@pl.jit` compilation method
- `run_prefill()` TQ branch — calls prefill_fwd_tq directly
- `run_decode()` TQ branch — calls decode_tq + _project_logits
- `get_compressed_block_info()` — generates cmp_block_table / cmp_num_blocks
- Scratch buffer allocation (tq_codebook, k_idx_scratch, v_idx_scratch)

## Verification

> **Status: Pending** — Results to be filled in after on-device testing.

- [ ] Single-prompt generation with `--kv_quant`: output quality and correctness
- [ ] Numerical comparison of logits: TQ on vs TQ off
- [ ] Multi-batch decode correctness
- [ ] Long-sequence KV cache capacity test
- [ ] Timing: prefill_tq vs prefill_fwd (BF16) latency comparison
- [ ] Memory: BF16 pages allocation verification

## Future Work

- **Residual window migration**: Periodically compress BF16 resident blocks into quantized format during decode.
- **BF16 allocation optimization**: If `decode_tq` supports dynamic stride, reduce BF16 pages to residual window size only.
- **Layer-adaptive precision**: `prefill_tq` currently uses fixed int4; future work could support per-layer bit-width.
