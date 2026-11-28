# V4.1 cache ABI migration

This work implements the coordinated cache migration described in
[serving issue #306](https://github.com/hw-native-sys/pypto-serving/issues/306).
This branch contains the ABI migration and the bounded replay implementation.
Startup rejects a Lib checkout that has not declared the complete new ABI.

## Logical blocks and storage

All scheduler block IDs represent 128 source tokens. Physical row counts differ:

| Family | Source tokens per block | Compression ratio | Storage rows per page |
| --- | ---: | ---: | ---: |
| Window | 128 | 1 | 128 |
| C2A | 128 | 2 | 64 |
| C1A | 128 | 1 | 128 |

For a completed compressed row at source position `p`, with compression ratio
`r` and physical page width `s`, addressing is:

```text
compressed_position = p // r
logical_block = compressed_position // s
row_in_page = compressed_position % s
physical_row = block_table[logical_block] * s + row_in_page
```

Both families therefore use `logical_block == p // 128`. C2A publishes only
completed pairs, at odd source positions. Incomplete pairs remain in a private
compressor state slot.

The packed compressed-KV and index pools use 64-row allocation units. Each of
the three C2A producers has one unit per logical page; the C1A producer has two.
The corresponding source views expose actual page shapes of 64 and 128 rows.
The two units of one C1A page are one allocation with one reference lifetime;
independent scheduler pages never share storage rows.

For `P` physical pages per family, including the dedicated padding page, a
compressed field occupies `P * (3 * 64 + 128)` rows. It does not allocate four
128-row sources after doubling the C2A logical page count. Page reset also uses
the source's actual page stride; resetting a C1A page clears both 64-row units.

## Compatibility boundary

Serving requires these literal constants in the Lib configuration and checks
the imported decode module again before compilation:

| Constant | Required value |
| --- | ---: |
| `BLOCK_SIZE` (or legacy `LOGICAL_BLOCK_SIZE`) | 128 |
| `C2A_STORAGE_ROWS` | 64 |
| `C1A_STORAGE_ROWS` | 128 |
| `COMPRESSED_POOL_UNIT_ROWS` | 64 |

The cache geometry is validated only after the affected prefill, decode, DSpark,
indexer, golden, and sliding-window consumers have migrated. Current Lib main
does not publish a standalone ABI version; Serving validates the geometry fields
at import time instead.

The Serving development base is `bbdf096`; the Lib development base is
`31bc47b4`. The Serving submodule pins merged Lib main commit `eac6fb1f`,
including the complete migration and unified logical-block geometry. The rebase preserves the refactored DSpark metadata,
initialization, and Markov paths. Serving binds the standalone generation head
after the normal decode backbone and calls the recurrent DSpark composite
directly. The V4.1 CI job pins PyPTO `3db18db1`, before PyPTO PR #3411.
The four-card measurements below were collected before that rebase, with Lib
`2cca3ea6`.
Device development uses four-card jobs.

## Bounded prefix replay and its accuracy limits

Issue #306 proposes excluding the window group from prefix restoration and
replaying 128 tokens from empty window and compressor state. This rebuilds a
window-shaped buffer, but does not generally reproduce its cold-run contents.

Consider two causal attention layers, each with a window of 128, uniform
attention weights, identity values, and a residual connection. Let input value
at position `p` be `p`, for positions 0 through 511. Evaluate the complete input,
then evaluate only its final 128 tokens from empty caches:

```python
def layer(values):
    return [values[p] + mean(values[max(0, p - 127):p + 1])
            for p in range(len(values))]
```

The last token's first-layer output agrees exactly: its complete 128-token
input window is available. The last token's second-layer output differs by
31.75, because earlier replayed first-layer outputs lacked their own preceding
windows. This is a mathematical counterexample to a general exact-restoration
claim, not an accuracy measurement of the checkpoint.

The selected policy follows the issue: acquire immutable compressed prefix pages
through boundary `H`, start a fresh private state slot at `H - 128`, and replay
128 tokens before computing the suffix. Replayed compressed KV and index writes
are disabled with `compressed_slots = -1` for every position before `H`.
No replay write is redirected into the padding page. Window, Engram, and
compressor state still compute and update normally. Attention reads use the
original shared compressed KV/index pages with unchanged page tables and
causal lengths. The resulting state deliberately combines newly computed
private rolling state with the original shared compressed history.

Replay and suffix share the normal prefill token budget. A chunk can start at
`H - 128` and end after `H`; the scheduler does not introduce a dispatch boundary
at `H`. Suffix positions at or after `H` publish to their private physical pages
with the usual C2A completed-pair rule.
Cold/warm numerical and answer differences must be measured on identical
prompts; bounded replay does not promise exact multi-layer state restoration.

The V4.1 CLI enables prefix caching with the 128-token replay policy. Shared
prefix pages must never be cleared or written by request reset, suffix
execution, abort, or release.

### Decoder replay configuration

Decoder Prefill bounded replay is controlled by
`RuntimeConfig.bounded_decoder_replay` (default `True`). The CLI exposes
`--bounded-decoder-replay` and `--no-bounded-decoder-replay`; the old
`PYPTO_DSV41_BOUNDED_REPLAY` environment variable is not read.

Each chunk computes only its intersection with the final 128-token Decoder
replay interval, clipping Decoder SWA reads at the same absolute replay
origin across chunks. For a 2056-token prompt split at 2048, Decoder work
is 120 rows followed by 8 rows. Encoder processing and C1A global publication
still cover every new row. DSpark bootstrap retains target taps in the
compact Decoder row order, including when a preceding chunk has no Decoder
rows. Set the runtime field to `False`, or use the negative CLI option, to
run full Decoder Prefill for numerical comparisons.

For example, enable bounded replay with DSpark K5 using
`--bounded-decoder-replay --speculative-config '{"method":"dspark","num_speculative_tokens":5}'`.

The V4.1 resident prefill graph is compiled for the per-request chunk limit.
The scheduler may use a larger `max_num_batched_tokens` budget to co-batch
multiple chunks without expanding every graph dispatch to that full budget.

### Publisher audit

Both compressed families pass the same write slots to their KV and index
publishers. Negative slots affect publication only; they do not mask reads or
the token's contribution to private state.

| Actual consumer | Negative-slot behavior |
| --- | --- |
| C1A KV: `c1a_compressor_publish` -> `kernels/compressor_publish.cpp` -> `publish_rows` | Normalizes and stores the unrotated latent for every active row, then skips both payload and scale stores when `slot < 0`. |
| C1A KV Python implementation: `_c1a_publish_body` -> `_c1a_publish_flush` | Assigns negative slots a token-based worker, computes/stores latent, and guards both cache stores with `target >= 0`. |
| C2A KV: `c2a_compressor_publish` -> `_c2a_compress` | Computes completed-pair latent before testing `target >= 0`; KV payload/scale stores are guarded. The request-partition state phase and token-partition state commit run independently of the slot. |
| C1A prefill index: `publish_c1a_index_key` -> `publish_index_cache` | Both index payload and E8M0 scale writes are inside `slot_i64 >= 0`. |
| C2A prefill index: `publish_c2a_index_key` -> `publish_index_key` | Both index payload and E8M0 scale writes are inside `slot_i64 >= 0`. |
| Decode index: `publish_index_rows` | Negative slots never enter its publication batch. |

C1A's decoder producer publishes from encoder rows before decoder replay
selection. Its reindex layers only read shared KV/index; reuse layers do not
publish compressed rows. The complete DP4 prefill uses the native C1A KV
publisher and token-partition C2A publisher above. No kernel change is needed
to implement the negative-slot replay policy.

## Validation completed so far

- Serving tests verify 128-token logical blocks, 64/128-row compressed pages,
  permuted physical page mapping across source-block boundaries, compact source
  strides, and rejection of incompatible ABI constants.
- The existing compressor golden entry now checks publication into compact
  pages as well as the retained speculative state ring. Four A5 cases start at
  positions 126, 127, 254, and 255. All pass with exact comparison of the complete
  packed cache, including untouched rows, and the existing latent/state checks.
- The complete Serving unit suite passed with 1,303 tests, including the
  replay, ownership, and geometry changes.

Broader accuracy evaluation across the target workloads remains required;
operator-level publication checks do not establish complete-model generation
correctness.

## Initial four-card validation and measured replay drift

The complete four-card DP4 programs (decode, prefill, and DSpark) compiled with
the initial Lib migration. Compact C2A operator cases at source positions 126,
127, 254, and 255 pass on A5; the long index cases at 126, 255, 1024, and
1290 also pass. The Serving unit set covering V4.1 geometry and grouped-cache
scheduling passes 105 tests.

A DP4 HTTP run with the 3,070-token arithmetic prompt completed successfully
with the cold answer `The product of 17 and 19 is 323.`. Three repeated warm
requests produced the identical answer and reused a 2,816-token compressed
prefix. The recorded terminal prefill logits had maximum absolute cold/warm
difference 3.43359375, mean absolute difference 0.50274885 (relative to the
cold mean absolute logit magnitude, 0.12981711). This is the expected measured
accuracy boundary of the selected 128-token replay policy; it is not an exact
state snapshot guarantee. Eight-request concurrent waves also completed with
correct arithmetic answers.

These measurements preceded the change from padding-page writes to negative
slots. They establish the initial bounded-replay behavior, not an accuracy
guarantee for every subsequent revision.

The HTTP guard now installs a test-only device snapshot probe. It compares
every shared compressed KV, index payload, and index scale page before and
after warm prefill, decode, and request reset. Prefill also checks the dedicated
padding pages. It asserts that a warm prefill dispatch crosses `H` and that
the cold, warm, and concurrent requests still return correct answers. The
probe writes its page records and terminal logits under the test temporary
directory for cold/warm comparisons, without adding a production runtime flag.

### Negative-slot replay validation

The DP4 HTTP guard passes on four A5 devices with Lib `2cca3ea6`
(target-only decode, `k0`). The
3,070-token prompt reuses a prefix ending at `H = 2816`; warm prefill runs from
2688 through 3069 in one 382-token dispatch. The next two warm requests also
share one 764-token dispatch in the same DP partition, exercising concurrent
readers of the same physical prefix pages.

The probe verifies 7,680 page snapshots across two warm prefill batches,
24 decode batches, and three request resets. All shared KV/index/scale bytes
remain unchanged, including the dedicated padding pages during prefill.
Each prefill checks 276 distinct pages: 264 shared pages and 12 padding pages.
Both eight-request concurrent arithmetic waves also pass.

All three warm answers exactly match the cold answer. The sequential warm
terminal logits have maximum absolute difference 3.43359375 and mean absolute
difference 0.50274885 from cold; the two co-batched warm rows have maximum
absolute difference 3.75 and mean absolute difference 0.62373775. All four
terminal rows select the same next token. These differences remain consistent
with the accepted mixed-state policy; answer equality on this prompt does not
establish general accuracy equivalence. Speculative `k5` execution was not
rerun for this follow-up.
