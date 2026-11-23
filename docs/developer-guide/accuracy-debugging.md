# Accuracy Debugging

Use this guide to investigate answer quality failures in Serving, record the
first divergence supported by evidence, and preserve reusable debugging cases.
Case results apply to their stated model, runtime, and workload; they are not
general accuracy or performance guarantees.

## Investigation workflow

1. **Freeze a reproduction.** Preserve the checkpoint, Serving and Lib revisions,
   local source patch, topology, runtime settings, exact prompt, token IDs, and
   generation settings. Keep original predictions and scores separate from
   diagnostic outputs.
2. **Inspect answers as well as scores.** Check whether the response addresses
   the actual question, whether an answer extractor happened to match a letter,
   and whether an output cap or empty final answer affected the grade.
3. **Reduce one variable at a time.** Compare the original input with simplified
   controls, vary context length and example placement, and check official prompt
   encoding before attributing the symptom to an operator or a version upgrade.
4. **Trace producer and consumer contracts.** Follow positions, page tables,
   source ownership, cache layout, quantized payloads, and scales across prefill
   and decode. In-bounds accesses and successful execution do not prove that the
   correct historical values were read.
5. **Find the first numerical divergence.** Compare exact inputs and intermediate
   state before comparing final answers. Distinguish byte comparisons from
   dequantized numerical comparisons, and document the comparison domain.
6. **Run a controlled treatment.** Hold weights and unrelated runtime behavior
   fixed. Explain any diagnostic wrappers and how their corrected behavior maps
   to the production fix. Report incomplete generations as incomplete.
7. **Validate the production path.** Follow the isolated reproduction with a real
   workload. Record coverage, sampling settings, execution failures, and any
   rebase or environment change that has not received device validation.

## Case record structure

For each new case, include the symptom, initial hypotheses, controls that ruled
out alternatives, confirmed cause, correction, validation, and remaining limits.
Separate observed facts from explanations that remain plausible. Keep generated
logs and prediction caches outside the documentation source tree; include enough
setup and measurements in the case to make the conclusion reviewable.

## Case 1: V4.1 index-key cache addressing

This investigation identified a Serving index-key cache layout mismatch that
caused a DeepSeek V4.1 request to reason about solved examples instead of the
actual question. Correcting decode addressing restored the correct answer in
the controlled reproduction. The observations below distinguish this confirmed
failure from the initial suspicion of a 1024-token chunked-prefill problem.

### Initial symptom and hypotheses

An early MMLU-Pro run scored 4/10, but seven responses discussed the RISC pipeline
few-shot example rather than their actual questions. Two scored hits were only
coincidental matches of the answer letter. These outputs were generated before
the Serving upgrade, so the failure could not be attributed to that upgrade.

The old GPQA-Diamond run scored 175/198 (88.38%), with 13 responses reaching the
32768-token output cap. All ten early MMLU-Pro prompts exceeded 1024 input tokens,
whereas 197/198 GPQA prompts did not. The configured prefill capacity was 1024,
making chunk boundaries, historical KV access, and context handling useful
initial hypotheses. This correlation alone did not establish a failure threshold.

### Narrow the failure before changing kernels

The diagnostic requests were saved separately from benchmark prediction caches:

1. **Check prompt rendering.** Serving's rendered input was byte-identical to
   the checkpoint's official encoder for all ten initial probe inputs.
2. **Check whether long inputs are universally truncated.** All 26 simple
   code-recall probes succeeded across 441 to 2041 rendered input tokens, with
   codes placed near the beginning, middle, and end. This ruled out universal
   truncation at position 1024, but did not prove general context correctness.
3. **Remove solved examples.** The computer-science target alone completed with
   the correct `ANSWER: C` in both chat and high-thinking probes.
4. **Keep one solved example.** A RISC example followed by the actual unification
   question still produced example confusion at 782 tokens in chat mode and
   808 in high-thinking mode. These bounded probes hit their 1024/2048 output
   caps. Crossing a prefill chunk boundary was therefore unnecessary for this
   symptom; these were not completed wrong-letter answers.
5. **Compare producer and consumer cache contracts.** Inspect both native
   prefill aliases and Lib's flattened decode tensors, including scale pools
   and reset ownership. An allocation can be large enough to avoid an address
   error while still yielding an incorrect source stride.

There were 56 live-service diagnostic requests in total. They contributed to
that service's aggregate counters, so those counters are not evaluation-only
performance measurements.

### Root cause: index queries are not index-key producers

The model has eight index-query/selection sources at layers
`2, 8, 14, 20, 24, 28, 32, 36`, but only four index-key producers at
`2, 8, 14, 20`. C1A reindex layers own independent query weights and selections
while borrowing layer 20's keys.

Lib's [decode implementation](https://github.com/hw-native-sys/pypto-lib/blob/b67a5fd60ccbc1e7451e8b995293c7f1bc76426b/models/deepseek_v4_1_flash/decode_fwd.py)
derives index-key pages per source by dividing the flattened page axis by
`KV_SOURCE_COUNT = 4`, rather than `INDEX_SOURCE_COUNT = 8`.
Serving instead allocated the key payload and scale axes using the eight query
sources. Native prefill published the first four key sources with stride G;
decode inferred stride 2G from the oversized flattened tensor.

For per-source physical page count G, the source base offsets were:

| Producer layer | Prefill offset | Incorrect decode offset | Correct decode offset |
| --- | --- | --- | --- |
| 2 | 0 | 0 | 0 |
| 8 | G | 2G | G |
| 14 | 2G | 4G | 2G |
| 20 | 3G | 6G | 3G |

Decode could consequently read another producer's keys or entries that prefill
had never published. All addresses remained within the allocation, allowing
requests to execute without a runtime error. The scale pool had the same defect.

A local CPU marker experiment published `[11, 12, 13, 14]` through the native
prefill view and read `[11, 13, 0, 0]` through the old decode view. Canonical
four-source allocation restored the expected markers. The mismatch also exists
in Serving revision `8d059b8`; it was not introduced by the upgrade to `ec8b063`.

The correction in [the V4.1 runner](../../pypto_serving/model/deepseek_v41/npu_runner.py)
allocates payload and scale pools by KV producers, assigns reset ownership by
those producers, and makes prefill reindex/reuse views borrow the KV producer's
keys. All eight index-query weight sources are retained.

### Controlled checkpoint A/B

The NPU experiment used Serving `ec8b063` plus the runtime fixes, Lib `b67a5fd`,
PyPTO `ed426adf`, and Simpler `355ce301`, on A5 with TP1/DP8/EP8. Its eight requests
ran sequentially with high thinking, temperature 0, top-p 1, seed 42, and explicit
512/1024 output caps. These deterministic settings differ from the benchmark's
temperature 0.95 and 32768-token output cap.

To isolate decode addressing, both treatments retained the same old eight-source
physical allocation, weights, compiled graph, and prefill bindings. The legacy
treatment exposed all eight sources as the decode tensor extent; the corrected
treatment exposed the first four, giving the canonical stride without moving
the buffers. This differs from production, which allocates only four sources.

One compiled prefill capacity of 2048 accommodated each full prompt. The
dispatcher either submitted it whole or cut it at 1024 tokens, preserving exact
input IDs, absolute positions, request identity, and page tables. Only the
terminal chunk sampled an output token.

| Input | Input tokens | Decode addressing | Prefill | Output tokens | Result |
| --- | ---: | --- | --- | ---: | --- |
| Computer science with examples | 1595 | Legacy | 1024 cuts | 1024 | Example confusion; output cap |
| Computer science with examples | 1595 | Corrected | 1024 cuts | 355 | Actual question answered correctly, C |
| Computer science with examples | 1595 | Corrected | Whole | 355 | Identical correct response |
| Computer science with examples | 1595 | Legacy | Whole | 1024 | Identical confused response; output cap |
| Computer science target only | 285 | Corrected | Whole | 336 | Correct, C |
| Computer science target only | 285 | Legacy | Whole | 330 | Correct, C |
| Math with examples | 1714 | Corrected | Whole | 512 | Actual integral discussed; output cap |
| Math with examples | 1714 | Corrected | 1024 cuts | 512 | Identical response; output cap |

The 285-token control used a different prompt wrapper from the earlier short
probes; its token count must not be substituted for theirs.

#### Find the first numerical divergence

Capture final prefill logits, request pages and positions, the first decode
logits, and these cache slices before generation changes them:

- 40 window cache fields;
- four compressed cache fields;
- four index-key payload fields;
- four index-key scale fields.

For each of the three whole-versus-chunk pairs, concatenated input IDs were
identical, final prefill logits were bit-identical with maximum absolute
difference zero, and all 52 cache fields were byte-identical. Dequantized
window/compressed caches also agreed at every layer, both over the full captured
history and the last row. Final responses were identical within each pair.

For both computer-science prefill settings, legacy and corrected decode
treatments also had identical input IDs, prefill logits, and cache bytes, even
on different DP partitions. The first decode step, at position 1595, already
differed: the top-16 sets overlapped on 15 IDs, with maximum absolute logit
difference 1.2417106628417969 on shared IDs. The first-step argmax still agreed;
later output trajectories diverged. This isolates the observed answer recovery
to decode addressing rather than changed input or prefill numerics.

The math response reasoned about the actual contour integral and obtained
`-pi*i/2`, then questioned the available options. It had no final answer within
512 output tokens and is not counted as a correct scored answer. This experiment
did not include a legacy math treatment.

### Related runtime fixes

Two other Serving fixes were present in both treatments and are included with
the cache correction. Their effects were not independently A/B tested here:

- **Logical history table width:** C2A and C1A index block table columns cover
  `ceil(max_seq_len / 256)` and `ceil(max_seq_len / 128)` respectively. Entries
  remain physical page IDs. Previously the column count used the entire physical
  pool capacity, mixing per-request history dimensions with resident allocation.
  CPU staging checks preserved physical page IDs across a prefill/decode boundary.
- **Explicit dependency pool configuration:** prefill now honors a supplied
  `runtime.ring_dep_pool` for DP8 as well as DP4, retaining the existing DP4
  fallback when unspecified. The validated configuration used heaps of
  `2147483648,2147483648,4294967296,4294967296` bytes and dependency pools of
  `32764,32764,32764,32764`.

### Production evaluation and limits

A subsequent run used the production four-source allocation, prefill capacity
1024 with chunking, maximum context 131072, 24 concurrent inference requests,
temperature 0.95, top-p 1, seed 42, high thinking, and a 32768-token output cap.
The HTTP client timeout remained 43200 seconds. It started from an empty
MMLU-Pro output directory and preserved the original five-shot prompts.

At the timed stop, 1573 of 12032 questions had saved predictions. Offline scoring
using the benchmark adapter gave 1394 correct (88.62% over the completed subset).
No Serving execution errors were recorded before shutdown. This is a partial
result, not a full MMLU-Pro score or a matched accuracy comparison with the old
ten-question run.

The evidence confirms the cache stride defect as a cause of the reproduced
computer-science failure. The tested prompts do not show a chunked-prefill
discrepancy. Context sensitivity is plausible because C1A selects 512 positions
and C2A selects 512 compressed positions corresponding to 1024 raw positions:
below a selection limit, incorrect key scores need not remove reachable history.
This is not a proof of a universal context-length threshold or an explanation
for every historical incorrect answer.

The NPU observations above precede the rebase onto current `main`; no NPU rerun
after that rebase is claimed. Raw prompts, predictions, captures, and service logs
remain external evaluation artifacts rather than documentation build inputs.
