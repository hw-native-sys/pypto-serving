# V4.1 bounded SWA segment

`pypto_serving/model/deepseek_v41/swa_segment.py` adds the concrete half-layer
execution boundary inspected against lib `216456332c2a74d89cca23b7824dab264ce34bff`:

1. `prefill_swa.make_hc_program(..., epochs=1)` includes mHC, input Norm,
   attention, TP communication and mHC post.
2. `moe.l3_moe` includes mHC, Norm, packed-FP4 routed/shared experts and mHC post.
3. The returned FP32 `LayerState` is passed directly to the next SWA layer.

The shared V4 `KernelCompiler` compiles both entries. `SwaSegment` uses one
persistent `DistributedWorker`, matching V4's runtime ownership. It does not
call small operators or copy intermediate state to CPU. The caller owns
uploaded weights, per-layer caches, positions, shared count metadata and scratch.
The example uses independent caches and tied synthetic weights for two layers.

Current bound: 16 physical token rows per rank, as fixed by lib `MOE_TOKENS`.
TP4 permits 64 rows per DP group; TP2 permits 32. Counts use contiguous slabs,
including padded ranks. Exceeding this capacity is rejected. Capacity checks and
count conversion do not establish numerical correctness of empty/ragged cases.
Device failures poison the segment; its worker must be closed before recovery.
Do not reset only the request cache and resume an uncertain collective.

Run the explicit synthetic diagnostic on A5 through the device queue:

```bash
PYTHONPATH=. python tools/validate_v41_swa_segment.py \
  --lib-root /path/to/current/pypto-lib --tp 2 --devices 0,1,2,3
```

`--compile-only` checks code generation without device execution. The device
smoke checks completion and finite/nonzero outputs, not numerical acceptance.
CPU dispatch tests use a mocked worker and do not establish NPU correctness.

This segment does not enable `load_composite_bindings()` for complete serving:
production resident bundle management, input initialization, all attention
modes, decode, cache lifecycle and the final output boundary still need adapters
and validation. Engram is excluded. Full TP4/DP2/EP8 8K-to-128 M0 is not claimed.

## Real checkpoint bundles and numerical diagnostic

`load_swa_layer_weights(model_dir, layer_id, topology)` returns CPU Attention
and MoE weight maps for a SWA layer. It uses the selective checkpoint loader;
TP projections and EP expert ownership retain their existing rules. Routed
payloads stay packed FP4. Per-expert scales must be unpacked and repacked into
the combined expert/K-group MX layout, not concatenated in their already packed
order. The caller still owns upload, request metadata and cache allocation.

To exercise distinct real weights for layers 0 and 1 and compare the final
state against composed Torch references:

```bash
PYTHONPATH=. python tools/validate_v41_swa_segment.py \
  --lib-root /path/to/current/pypto-lib --tp 2 --devices 0,1,2,3 \
  --model-dir /path/to/DeepSeek-V4.1-Flash --reference --ring-heap-mib 4096
```

The default `--input-source stress` uses controlled random HC activations and
request metadata, including in checkpoint mode. `--input-source embeddings`
instead loads checkpoint embedding rows, broadcasts each row to four HC lanes,
and uses an identity pre-mix selecting lane zero. By default it selects sequential
token IDs. Supply `--token-ids /path/to/ids.json` for an integer JSON array shaped
`[DP, capacity]`, such as tokenizer-produced full token slabs. IDs are not padded,
repeated or truncated by the diagnostic. Both controls retain fixture RoPE/pages;
neither establishes complete prompt semantics, Engram or generation. The selected
IDs and exact initial state are saved for CPU replay. This diagnostic input setup
does not enable the production composite adapter.

The reference runs only after the device worker
closes, never supplies intermediate device inputs, and shares read-only weights
with the device path. Final actual/expected residual and pre_mix are saved to
`--artifact-dir/comparison.pt` even when the numerical comparison fails. The
residual gate defaults to `--residual-profile dsv4-layer`: the DSV4 complete-layer
comparator `ratio_reldiff(diff_thd=0.01, pct_thd=0.05)`, applied separately to each
rank. `--residual-profile v41-local` retains the historical V4.1 single-MoE
comparator (0.003/2%, with its single-point cap). Accumulated pre_mix uses the user-approved provisional rtol=0.01 and
atol=0.001, requiring every element to pass; local stage gates are unchanged.
This is limited to this two-layer diagnostic, not a V4/vLLM multi-layer standard. A completed smoke is not a numerical pass.

The real-weight TP2/DP2/EP4 diagnostic exhausted the temporary heap at both
512 MiB and 1024 MiB per ring. With 4096 MiB per ring, both device layers and
their Torch references completed. This runtime has four rings, so a per-ring
setting is not the total allocation; check device headroom before running.
The saved final outputs did not pass the numerical gate (residual relative L2
about 0.00760, pre_mix about 0.00105). These are diagnostic observations, not
an accepted end-to-end tolerance or a production memory recommendation.

`--stage-reference` also captures each half-layer's outputs after the entire
device chain completes, and applies lib's stage comparators to references
computed on that half-layer's actual input. This localizes accumulated errors;
it does not replace the independent end-to-end check or feed CPU values back
to the device. The final check still determines success.

At serving `7ef97ca`, all stage checks passed for both real-weight SWA/MoE
layers, including Attention caches and MoE residuals. The independent two-layer
check still failed with the errors above. Agreement on each stage's actual input
does not establish the accumulated numerical budget across layers; that boundary
still needs validation before full-model acceptance.

Under the historical `v41-local` profile, the baseline embedding controls fail the accumulated gate, while
their 18 native half-layer checks pass:

| Initial state | Rank-zero outliers | Residual relative L2 | Pre-mix relative L2 |
| --- | ---: | ---: | ---: |
| Independent random HC streams | 32.572% | 0.00760 | 0.00105 |
| Sequential checkpoint embedding rows | 3.896% | 0.01316 | 0.00281 |
| Checkpoint embeddings for text token prefixes | 4.094% | 0.01348 | 0.00284 |

The text control uses 32 tokens per DP group with fixture metadata. The lower
outlier fraction does not mean relative L2 improved: these are distinct
measurements on distinct workloads. The original stress failure remains an
unresolved regression case. On the sequential embedding control, CPU replay
matches the saved full reference bitwise; accumulated residual relative L2 at
Attention 0, MoE 0, Attention 1, and MoE 1 is 0.00101, 0.00353, 0.01029, and
0.01316 respectively. CPU router replay on inherited device inputs changes four
second-layer expert sets in the random control and none in this embedding
control. Routing changes alone therefore do not explain the chain failure.

For precision bisection, follow lib's `docs/debug-and-tune/precision-tuning.md`
and PyPTO's `docs/en/user/precision/00-workflow.md`. Check dtype, rounding and
reference operation order before changing kernels. The historical full-chain gate
reused a single-MoE comparator; it was not an agreed model-wide error budget.
Report relative L2, maximum absolute error and outlier fraction separately, and
retain the historical results when changing acceptance profiles.
FP8 trace comparisons decode payloads with their own scales; different encoding
pairs can represent identical values. Local checks on actual device inputs and
CPU boundary substitutions only localize errors, never replace full-chain
acceptance or supply intermediate values to device execution.

Rechecking those same saved tensors with `dsv4-layer` passes residual on every
rank. Worst-rank outlier fractions are 3.3542% (random streams), 0.02167%
(sequential embeddings), and 0.12879% (text embeddings), below the 5% budget.
This is a requested acceptance-profile change, not reduced numerical error.
With the historical pre_mix atol=0.0001, 14/256, 30/256 and 24/256 entries
respectively failed, so that historical overall diagnostic failed. DSV4's layer comparison is not an independent
43-layer accuracy guarantee, and it does not specify V4.1's delayed pre_mix
contract. These results do not establish complete model or M0 acceptance.

To recheck a saved final comparison without compiling or allocating devices:

```bash
PYTHONPATH=. python tools/validate_v41_swa_segment.py \
  --lib-root /path/to/current/pypto-lib --tp 2 --devices 0,1,2,3 \
  --compare-only /path/to/comparison.pt
```


On 2026-09-28 the user approved provisional accumulated pre_mix tolerances
rtol=0.01, atol=0.001, with no allowed failing elements. CPU rechecks of the
three saved device runs still fail in 2/256 (random), 7/256 (sequential
embeddings), and 7/256 (text embeddings) entries. Residual and all native stage
checks pass, but overall two-layer acceptance remains unresolved. This changes
the acceptance budget only; numerical errors are unchanged. No new NPU run was
performed. Evidence: `.validation-artifacts/approved-premix-budget-recheck.json`
on the A5 validation checkout, using baseline lib `21645633`.


`segment_inputs.prepare_segment_inputs` prepares fresh host HC input buffers
from packed BF16 embeddings and a `ForwardStep`. Like V4 host input preparation,
it maps request rows before device upload. The initial four FP32 lanes and
lane-zero pre-mix follow lib `input_pack.pack_x_hc` and `golden.identity_pre_mix`
at `fbe92bfc`. This is data packing only; it does not run mHC or normalization.

Requests may interleave DP partitions. Each partition keeps its own stable
packed order, split into contiguous TP token slabs. Returned source-row indices
also define the mapping for positions and other token metadata; returned final
row locations preserve the original request order for later output selection.
Inactive rows have zero residual/pre-mix, including fully empty DP partitions.
The helper rejects over-capacity steps before allocation instead of truncating.
It is used by the embedding diagnostic; cache lowering, device upload and the
complete production adapter remain separate work. Never invoke it between layers
to overwrite the residual/pre-mix produced by the previous composite.


`swa_metadata.prepare_swa_window_metadata` lowers scheduler-owned full-history
128-row window pages into the lib's INT64 write slots and INT32 causal read
indices. TP peers receive identical metadata within each DP group; physical
page IDs may be reused across DP groups but never shared by active requests in
one group. Each query reads at most 128 positions ending at itself. Full-history
pages avoid overwriting an early query's history when an entire new chunk is
published before attention. Rolling/modulo page reuse is not implemented.

The metadata helper covers page crossing, chunk continuation into decode,
interleaved requests, padding and empty partitions. It does not allocate or clear
the cache or choose a RoPE profile. Its positions must select the corresponding
checkpoint RoPE rows before dispatch. Integration into a complete model adapter
and device validation of that lifecycle remain pending.

`gather_swa_rope_rows` selects FP32 rows from the lib-built SWA tables using
those absolute positions, with identity rotation for padding. Like V4's
executor, model-specific table generation stays in the selected lib helper.
The diagnostic's optional `--request-metadata` requires embedding inputs and
a checkpoint. It uses fresh private request pages, empty caches and checkpoint
RoPE instead of fixture history/angles, saving all request inputs for exact
CPU replay. It currently exercises one full first chunk per DP group; it is
not a continuation, generation or complete backend acceptance test. Keep the
historical fixture failures separate from results on this changed workload.

`prefill_segment.PrefillSegment` extends the same device handoff to the audited
C2A Full/Reuse and C1A Full/Reindex/Reuse SP entry signatures. It calls existing
lib composites, with caller-supplied weights, metadata, scratch and caches.
C1A uses `prefill_c1a_sp.make_program`, preserving TP-local residuals. Each
compiled Attention program has its own retained-window epoch, while one shared
MoE program advances every layer. Missing arguments are rejected before either
half-layer runs. A failed dispatch poisons the entire worker.

This dispatch support is not full-model readiness: real compressed-cache
allocation, complete request integration and device numerical validation remain
required. `tools/compile_v41_prefill_segments.py --lib-root ... --modes c2a_full`
provides an explicit compilation-only check without allocating devices. All
programs must be registered with the same persistent worker before execution.

`compressed_metadata.prepare_compressed_metadata` prepares C1A/C2A positions,
causal compressed lengths, publication slots, index page tables and stable
compressor state IDs from a `ForwardStep`. It requires jointly numbered KV/index
pools, as current composites publish their index key at the compressed KV slot.
Ratio-2 publishes at odd positions and rotates at the pair's first position;
request state IDs survive batch reordering. Top-K/candidate selection remains
inside lib, and producer buffers stay caller-owned. Host tests cover pair/page
boundaries and continuation; this does not establish device lifecycle readiness.

`PrefillSegment.run_chain` binds resident producer handles before dispatching
consecutive layers for one packed request step. Full layers own compressed KV
and index caches; Reindex consumes the earlier candidate mask and produces its
own Top-K; Reuse consumes that index producer's physical Top-K rows directly.
Every layer keeps its own SWA cache. C1A's unused common-ABI weight slots refer
to real producer weights, without loading nonexistent Reuse weights. Each
producer must appear earlier in the same chain, preventing stale transient
selections from a previous step from satisfying an omitted producer. All
argument names and compiled modes are checked before the first dispatch.
The caller still owns allocation, step metadata, reset and buffer lifetime.
Unit tests cover the checkpoint's 40-layer producer plan; this is not evidence
of 40-layer device execution or numerical acceptance.

`tools/validate_v41_c2a_chain.py` continues a saved fresh-request embedding
diagnostic through checkpoint layers 2/3 (C2A Full, packed-FP4 MoE, C2A Reuse,
packed-FP4 MoE). It uses the request metadata helpers, empty layer-owned caches,
checkpoint-compatible lib RoPE tables and resident producer bindings.
`--prepare-only` checks real-weight host preparation without allocating devices.
The device path retains each stage's outputs and applies the existing lib
same-input stage comparators after worker shutdown. A native-stage pass does
not resolve the incoming SWA accumulated error or establish a new accumulated
four-layer acceptance standard. Source state and numerical artifacts must be
reported with that limitation.

The fresh real-weight C2A pair passed all 24 native checks on A5 TP2/DP2/EP4
at serving `9f9a5c3` and official lib `fbe92bfc`. At serving `fabf011`, the
`--group-counts 31,0` case also passed, covering an odd compressor tail and an
empty DP group with untouched caches. Both device tasks exited zero and released
all four cards. These remain stage/state checks with an unresolved incoming
SWA accumulated error.

`--continue-to-capacity` adds a second chunk for each initially nonempty request,
using the remaining rows of the saved causal source. For example,
`--group-counts 31,0 --continue-to-capacity` runs 31 then 1 token in the first DP
group, with the second group empty. Both chunks use the same resident per-layer
caches, compressor state, weights and communication windows. Step snapshots are
read-only diagnostics; references execute after worker shutdown. This checks
C2A continuation, not SWA chunk equivalence, full-model accuracy or request reset.

That `31 + 1` continuation passed all 48 native stage/state checks on A5
TP2/DP2/EP4 at serving `0355d40` and official lib `fbe92bfc`. Task
`task_20260929_040845_198102410502` exited zero and released cards 0-3.
Evidence: `c2a-chain-continuation-retry/comparison.pt` and
`c2a-chain-continuation-retry.log` under the validation artifact directory.
An earlier attempt stopped before device execution with an empty PTOAS error;
the identical generated O-B source compiled in isolation. The successful retry
explicitly limited `PYPTO_CODEGEN_MAX_WORKERS=4` as well as build/OMP workers.
No source, reference or acceptance threshold changed between these attempts.

The minimal Q-B group-32 diagnostic candidate (`cd759ce7`, based on official
lib `fbe92bfc`) was also tested with the same real-text inputs and explicit
request metadata, using serving `6f52516`. All native checks and all per-rank
V4 residual checks passed, but accumulated pre_mix still failed: 3/256 elements,
relative L2 0.0025390928 and maximum absolute error 0.0076903105. Residual
relative L2 was 0.012873608 (maximum absolute 0.02734375). The baseline has
7/256 pre_mix failures on this workload. This is an improvement, not an accepted
precision fix; the candidate remains on a diagnostic branch and is not a
production dependency. Neither reference arithmetic nor gates were changed.

The Q-B plus O-B group-32 diagnostic (`0e166cd9`) also completed the same
real-text/request workload. Residual passed on every rank (relative L2
0.012825638, maximum absolute error 0.02734375), but pre_mix still failed in
2/256 elements (relative L2 0.0022762596, maximum absolute error 0.0088607967).
The smaller failure count is not acceptance: its maximum absolute error is
higher than the Q-B-only candidate. It remains diagnostic, with no lib pin or
reference change. Evidence: `swa-realweights-qb-ob32-text/comparison.pt` and
`swa-precision-qb-ob32-text.log` under the validation artifact directory.

The request ledger retains committed full-history pages for omitted/paused
requests until their reset succeeds. A later batch cannot borrow those pages,
and a continuation may only append to its committed page table. Relocation or
dropping history is rejected because no device cache-copy contract is integrated.
Page IDs remain independent across DP partitions and cache groups. These host
ownership checks complement per-batch metadata validation; device reset/reuse
still requires its own validation.

### TP communication precision isolation

At serving `0355d40`, diagnostic lib `be5609d5` captures the O-B partial,
publication input, actual peer reads and pre-cast reduction. The producer and
publication inputs are bitwise equal. Four peer-read values across two layers
differ near the last row's final columns. Addition and BF16 conversion match
the captured reads, but do not always match the values published by the peers.
For layer 0, DP group 1, row 31, column 5115, the published FP32 partials sum to
1.0050979852676392 (BF16 1.0078125); the device returns 1.0. Instrumented final
outputs are bitwise identical to the uninstrumented Q-B group-32 diagnostic.

The fixed-partial, reduction-only replay (`ea9e9a7c`) passes exactly. Adding
the input AllGather and its communication windows (`9061b7e0`) reproduces that
output mismatch while the AllGather output remains bitwise exact. This replay
loads captured partials, not checkpoint weights. Disabling the output
publication pipeline in the full workload (`0e524332`) does not change the
failure. Evidence includes `tp-gather-replay-layer0.log`,
`tp-gather-replay-layer0.pt` and `swa-realweights-tp-read-trace/` in the
validation artifact directory. All completed tasks released cards 0-3.

The four-window replay still fails when AllGather execution is omitted.
Reserving 64 bytes for each signal makes it pass without changing arithmetic.
The original 32-byte allocation padding places an output payload tail and its
signal on one 64-byte scalar cache line; signal cache maintenance can affect
the neighboring payload. Existing tracking: `hw-native-sys/pypto#2800` and
`hw-native-sys/simpler#2273`.

PyPTO diagnostic fix `b792bde6`, based on `e8191e3c`, rounds both each physical
buffer size and their summed window capacity to 64 bytes. Logical views and the
runtime allocation ABI stay unchanged. The original 8-byte logical signal
replay now passes with zero mismatches (`task_20260929_101804_154960424725`).
The full real-weight trace (`task_20260929_102020_17364556272`, lib `d0478d78`,
serving `0355d40`) also has zero publication/peer-read, sum or cast mismatches
across both layers and all four ranks. Both tasks released cards 0-3.

This fixes the captured communication corruption, but accumulated pre_mix
still fails in 3/256 elements: relative L2 0.0025418127, maximum absolute error
0.0076903105. Residual passes every rank's V4 gate (relative L2 0.012851512,
maximum absolute error 0.02734375). Reference outputs remain bitwise equal to
the earlier Q-B-only run. No arithmetic or acceptance threshold changed.
Artifacts: `tp-codegen64-replay-layer0.log`, `swa-precision-codegen64-trace.log`
and `swa-realweights-codegen64-trace/tp-boundary-audit.json`.
The compiler fix and Q-B group-32 arithmetic remain diagnostic dependencies;
these results do not establish full-model precision or M0 acceptance.

### C1A diagnostic preparation

The compressed-chain diagnostic also accepts `--family c1a`. It resolves the
checkpoint's first C1A Full producer and executes consecutive layers through
`--last-layer` (inclusive). The default covers layers 20/21. Using
`--last-layer 25` retains producer 20 through Reuse layers 21-23, then executes
Reindex 24 and its Reuse consumer 25. KV/index caches, candidates and Top-K
buffers are bound to their declared producers, including read-only integrity
checks. Weight loading uses the same selective loader as the C2A path.

For example, add `--family c1a --prepare-only` to the compressed-chain command
to check real weights and metadata before device execution. This mode uses
the saved SWA state as an explicitly injected diagnostic input; it does not
claim to have executed layers 2-19. Artifacts record the chosen family, layer
IDs and this limitation. The existing native C1A comparators are preserved.
Host plan/metadata tests and real-checkpoint preparation pass. On A5 four-card
TP2/DP2/EP4, serving `a11532c`, official lib `fbe92bfc` and the previously
validated PyPTO communication fix `b792bde6` execute both layers and pass 28/30
native checks. Full layer 20 fails `attn_output` and `x_hc_out`: DP group 1,
row 18 has attention error RMS 0.0090870445 against limit 0.0083356445.
Same-input HC post replay is within its local budget, but end-to-end residual
relative L2 exceeds the native 1% limit on ranks 2/3. No comparator changed.

Full-layer cache, candidate and Top-K checks, all Reuse-layer checks (including
read-only producer state), and both MoE stages pass. The earlier task rejected
aliased index ABI arguments before executing Attention; serving now keeps
unused index slots disjoint and shares only the actual producer selection with
its consumer. The runtime rejection is resolved; the numerical failures remain.
Evidence: `c1a-chain-disjoint/comparison.pt`, `c1a-chain-disjoint.log`, task
`task_20260929_113608_405763621400` (exit 1, all four cards released).
Later Reindex and continuation results appear below. This is not accumulated
full-model or M0 acceptance.

The tag-only C1A capture (`b4fa1c91`, serving `14e6a5a`) leaves every saved
actual and expected stage tensor bitwise unchanged. Same-input QNorm, query
RoPE and inverse RoPE are exact; both DP groups' captured TP sums also match
exactly. Q-A, Q-B and O-A relative L2 errors are on the order of 1e-5, and O-B
is about 2e-7. A CPU replay of the kernel's BF16 probability narrowing
reproduces over 99.998% of the captured core elements. Propagating the captured
core through the independent output projection reduces output relative L2 to
0.00012 / 0.000047. These cuts are diagnostic, not acceptance results.

The initial attention-core audit mistakenly used the generic FP32-probability
reference: its 0.00151-0.00165 relative L2 is not the native C1A contract.
All three modes inject `golden_prefill_c1a_attention`, which includes BF16 PV
and the first-vector FP32 patch. The probability-residual candidate `1d186943`
is rejected: it passes only 27/30 native checks and introduces a Reuse output
failure. First-layer expected tensors are bitwise unchanged from baseline;
the candidate never changed the acceptance reference or gates. The initial
`abfa7197` attempt stopped at accumulator dtype checking; the retry explicitly
declares FP32 and completed device execution before failing precision.
Further bisection must follow the specialized reference, not replace it with
generic sparse attention. The same caveat applies to SWA's BF16-P reference.
Evidence: `c1a-chain-trace/boundary-audit.json`, `boundary-audit.pt`,
`cpu-boundary-cuts.pt` and `c1a-cpu-boundary-cuts.log`.

With the specialized reference, the original DP1 token 18 attention error
is 1.0903%; continuing from captured Q-A reduces it to 0.1855%. Independent
FP64 evaluation confirms a Q-A BF16 rounding error at that token. Candidate
`b80bb8b3` routes all three C1A modes through the existing scale-corrected
group-32 prefill Q-A implementation. On the same Full20/Reuse21 case, it
passes **29/30 native checks**, including the previously failing Attention
output. Only Full20 end-to-end `x_hc_out` fails: ranks 2/3 relative L2 are
0.0161307 / 0.0119976. All first-layer expected tensors remain bitwise equal
to baseline. No probability arithmetic or acceptance check changed.

Query/cache boundary cuts isolate this remaining error: using captured caches
with the canonical query reduces HC relative L2 to 0.003437 / 0.002168;
changing only the query does not improve it. The canonical CPU replay matches
the saved expected Attention output exactly. This is diagnostic evidence,
not a replacement reference or a passing full-layer result.

The original normalized input differs in only six BF16 elements. At all six
coordinates, device values match the independent FP64 collapse/normalization
rounded to BF16. For example, DP1 token 0, feature 4237 has an exact FP64
collapse sum of 0.252929660224396; the FP32 reference lands on the BF16 midpoint
0.2529296875 and rounds to the other neighbor. Subsequent quantization changes
cache values and amplifies the difference. This does not establish an incorrect
device HC implementation. The original gate remains failed; neither weakening
it nor changing the reference is part of this diagnostic. Evidence:
`c1a-chain-qa32/comparison.pt`, `query-cache-cuts.pt`, `c1a-chain-qa32.log`,
`c1a-query-cache-cuts.log` and `c1a-input-rounding-details.log`.

The six-layer C1A chain (20-25, MoE after every Attention) completes with
**88/90 native checks passing** on the same Q-A candidate and compiler fix.
Full20 HC remains failed. Reuse22 additionally fails Attention at DP0 row 2:
error RMS 0.013177692 versus limit 0.010058112, and peak 0.056640625 versus
limit 0.050295558. Reindex24, Reuse25, all cache/selection integrity checks
and all MoE stages pass. Task `task_20260929_124358_401077729544` exits 1
and releases all four cards; evidence is `c1a-through-reindex/comparison.pt`.

C1A Full20/Reuse21 also completes **31+1 continuation with an empty DP
group**, using the same resident cache and communication allocations.
All **60/60 native stage/state checks pass**. Task
`task_20260929_125339_40626412299` exits 0 and releases cards 0-3; evidence
is `c1a-chain-continuation/comparison.pt` and its adjacent task log.
Both cases use serving `14e6a5a`, lib candidate `b80bb8b3` and compiler
`b792bde6`. This remains bounded four-card evidence, not accumulated or M0
acceptance.

`tools/replay_v41_c1a_attention.py` isolates a Reuse Attention composite from
a saved single-chunk chain. It loads Attention weights without routed experts,
restores the preceding MoE output and actual producer caches/selections, and
checks bitwise agreement with the original actual and expected stage tensors.
Only after equivalence passes should optional tagged captures be interpreted.
The replay is diagnostic and does not replace independent accumulated gates.

Reuse22 isolated replay (`9ad837e` / tag-only lib `da929c70`) reproduces all
actual and expected stage tensors bitwise. The native row-2 failure remains.
Its native-reference Q-A boundary cut reduces that row's relative L2 from
0.01310287 to 4.24e-7. Independent FP64 confirms a Q-A accumulation rounding
error at DP0 row 2, column 308: device -0.0252685546875 versus
reference/FP64-rounded -0.025390625.

A compensated Q-A accumulation candidate (`e537ef47`) corrects this row,
but still fails the original native Attention gate at DP0 row 31 (RMS
0.011790978 versus limit 0.0093322441). All other 11 checks pass and every
original expected tensor remains bitwise unchanged. This candidate is not
accepted. A subsequent Q-A-only device capture (`e7d72872`) leaves every
actual and expected tensor bitwise unchanged from the untagged candidate.
Captured Q-A relative L2 against independent FP64 is 2.18e-8 in DP0 and
zero in DP1, versus about 2.23e-5 / 2.27e-5 for the native FP32 reference.
At DP0 row 31, column 722, the FP64 sum is 0.09545897599309683; device and
FP64 round to BF16 0.09521484375, while the reference rounds to 0.095703125.
Continuing from captured Q-A reduces that row's output relative L2 from
0.0126360 to 0.00122239. This isolates amplification of a reference/device
accumulation-rounding difference; it does not make the original acceptance
pass. Neither the independent reference nor its gate was replaced.
Evidence: `c1a-reuse22-kahan/comparison.pt`,
`c1a-reuse22-kahan-trace/qa-captures.pt`, `qa-propagation-cuts.pt`,
`c1a-reuse22-kahan-device-fp64.log` and `c1a-kahan-qa-propagation.log`.

Candidate replay uses explicit `--candidate`: it requires unchanged original
expected values and all native checks to pass. Default replay exit zero
means only equivalence to the original run, whose numerical failure may
remain. Neither mode replaces an accumulated acceptance test.

An additional control (`5b875e2a`, Full-only prefill Q-A change) restores
Reuse22's official original Q-A implementation on the same frozen input.
It still fails row 31 with exactly the compensated candidate's error RMS
0.011790978 and limit 0.0093322441. Every saved actual and expected tensor
is bitwise equal to the compensated candidate. Thus reverting Reuse Q-A
does not remove this native gate failure. Task
`task_20260929_140153_32103113893` exited 1 and released cards 0-3;
evidence is `c1a-reuse22-original-qa/comparison.pt` and
`c1a-reuse22-original-kahan-equivalence.log`. No candidate is promoted.

### Bounded cross-page continuation diagnostic

`validate_v41_c2a_chain.py --repeat-input-chunks N` repeats the saved injected
boundary input at advancing absolute positions, retaining device caches and
communication windows. It accepts 2-16 full chunks, separately recording the
source offsets and request positions. It cannot combine repetition with ragged
nonempty counts or `--continue-to-capacity`; empty DP groups remain inactive.
The source is not output from a complete
preceding model segment at those positions; this is a state/ABI diagnostic.

At TP2, five 32-token chunks cross a C1A 128-row compressed-cache page;
nine chunks cross a ratio-2 C2A compressed page. The window cache, KV/index
pools, candidate mask and RoPE extent cover the full diagnostic context.
Native comparisons use the previous chunk's captured cache state and the
current chunk's producer outputs. There is no intermediate host feedback
into the device chain. Host tests cover disjoint successive write slots,
causal cross-page reads, compressed lengths and physical page extents.
This option does not establish 8K prefill, reset/reuse, full-model numerical
accuracy or M0 acceptance.

The A5 C2A single-request control at serving `4b39b38`, official lib
`fbe92bfc` and PyPTO `b792bde6` completed nine 32-token chunks. Its
36 Attention/MoE stages passed **216/216 native checks**; the last chunk
covered positions 256-287. Saved-state audit confirmed that the second
compressed page contains published payload, the other DP group's cache
is untouched, and all earlier chunk caches remained resident. Task
`task_20260929_143008_89019724616` exited zero and released devices 0-3.
Evidence is `c2a-crosspage-single-request/comparison.pt`, its adjacent
task log and `c2a-crosspage-audit.log`. The input was repeated from the
saved boundary state, so this is a physical-page continuation and native
state check, not an independent accumulated model run.

The A5 C1A Full20/Reuse21 control at serving `52cb5f9`, diagnostic
Full-only Q-A lib `5b875e2a` and the same PyPTO revision completed five
32-token chunks. All **150/150 native checks** passed across 20 stages;
the last chunk covered positions 128-159. Its second window, compressed
KV and index pages contain published payload, while the inactive DP group's
caches remain untouched. Task `task_20260929_144014_160907026376` exited
zero and released devices 0-3. Evidence is
`c1a-crosspage-single-request/comparison.pt`, its task log and
`c1a-crosspage-audit.log`. This one-active-DP diagnostic does not supersede
the prior two-active-DP Full20 HC and Reuse22 failures, nor prove that
layers 0-19 supplied the injected boundary input.
