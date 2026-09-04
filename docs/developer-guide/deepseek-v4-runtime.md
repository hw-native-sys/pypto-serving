# DeepSeek V4 Runtime

DeepSeek V4 uses a model-local eight-rank runtime path. Attention data parallelism and MoE expert parallelism reuse the same physical ranks, so `--dp 8 --ep 8 --tp 1` is one overlapped model replica rather than eight independent serving replicas.

## Dynamic Prefill Extent

The main prefill wrapper supports four requests per DP partition, each with a dynamic extent up to 8192 tokens, and walks every request internally in 128-token tiles. DP=8 therefore exposes 32 prefill request slots per dispatch. The leaf attention and MoE programs remain B1/S128 and reuse their bounded workspace across request slots. AR and MTP use the same per-request limit; the effective extent is the minimum of 8192, `--max-num-batched-tokens`, `--long-prefill-token-threshold`, and `--max-model-len`.

AR and MTP submit each main-prefill chunk once, with its backing extent padded to the next 128-token tile. An 8191-token prompt therefore uses one 8192-row main-prefill dispatch when those configured limits permit it, rather than 64 separate serving dispatches. The 128-token width is an internal kernel tile, not a serving chunk restriction.

The main kernel returns each owner's final 128 valid pre-HC rows, which the standalone fixed-width MTP prefill kernel uses to rebuild its 127-row KV window before retaining the final row for sampling.

## Cache Layout

DeepSeek V4 uses grouped cache pools instead of the generic KV cache tensor layout. The seven main-model KV/state pools are allocated during runner preflight as rank-sharded worker-resident tensors.

Prefill and decode pass the same device handles and address them with scheduler-owned group block IDs. There is no prefill CPU snapshot or cache handoff. Reassigned pages are cleared with targeted host-to-device copies before their new owner writes them.

## MTP State

MTP prefill context, draft token, recurrent hidden state, and acceptance counters are owned by request ID. MTP prefill and decode share one worker-resident cache, but each request addresses it with scheduler-owned rank-local `ori` block IDs.

The scheduler reserves all speculative positions before dispatch, including when a draft sequence crosses a 128-token page boundary.

Before the first decode is prepared, each request owns a stable rank-local device-state slot and reuse generation. Terminal prefill fills that reserved slot with the committed tail token, next draft token, tail position, and committed count.

The fused decode kernel uses `(rank, slot, generation)` to build the next `[tail, draft]` input rows and sequence metadata before main decode, then updates the same slot after MTP verification. Host output processing mirrors the state for scheduling and statistics, but is not an input dependency of the next steady-state decode.

Generation matching prevents a stale queued step from updating a slot after preemption and reuse.
