---
name: adapt-new-model
description: Adapt pypto-serving to a model checkpoint it does not yet serve — set up the pypto env stack, triage how deep the change goes (loader-only vs new model family module), implement family detection, weight loading, staging spec, executor/runner, CLI wiring, tests, and finish with NPU device validation. Use when asked to add, port, or adapt a model to this serving stack, including new Qwen or DeepSeek checkpoints, new quantizations, or a new architecture.
---

# Adapt a New Model

Adding a model is a code-integration task, not a config-file change. A model
touches the env stack, family detection, the loader registry, a weight-staging
spec, an NPU executor and runner, CLI dispatch, topology validation, tests,
and docs. This skill fixes the order, names the reference implementation for
each piece, and routes the device pitfalls that recur on every adaptation.

Ground rules:

- Do not hard-code a specific model path, device id, or user directory. Use the
  checkpoint and device budget the user provides, and print what actually ran.
- Do not modify `model/common/` when a family-local file will do; conversely,
  do not fork shared machinery that only needs a new rule or parameter.
- Nothing is done until it generates coherent text on an NPU.

References in this skill directory:

- `references/env-stack.md` — installing, refreshing, and switching the pypto
  runtime stack (pypto / simpler / pto-isa / ptoas). Read it before any device
  reproduction.
- `references/device-fault-triage.md` — symptom-to-cause table for device
  faults with the exact log signatures. Read it before debugging any device
  failure; most "new-model bugs" are one of its nine rows.

## Phase 0 — Environment: the pypto Stack

The serving repo pins kernel sources (`pypto-lib` submodule), but device runs
execute against a compiled host stack (pypto, simpler, pto-isa, ptoas)
installed in a conda env. Stack skew produces faults that look exactly like
model bugs, so settle the environment first:

1. Confirm the worktree's submodules are initialized:
   `git submodule update --init --recursive` (fresh worktrees start empty).
2. Refresh the stack to the pins the user targets — the full procedure,
   verification, and the silently-stale-simpler trap are in
   `references/env-stack.md`. After a refresh, the stack sanity gates
   (pypto's qwen accuracy and JIT-cache tests) must pass on one card before
   any new-model debugging is trusted.
3. Note the stack's provenance in the final report (pypto commit, simpler
   commit, ptoas version) so device results are reproducible.

## How a Model Is Wired Through the Stack

The dispatch chain every adaptation extends, from CLI to kernel:

1. `pypto-serving --model <dir>` parses args in `pypto_serving/cli/main.py`.
2. `read_model_config` + `detect_model_family` (`pypto_serving/model/model_family.py`)
   turn the checkpoint's `config.json` into a family string.
3. `build_serving_engine_config` (`pypto_serving/cli/main.py`) picks the
   executor class id string (`_executor_cls_for_model_family`), the placement
   mode (`replica` for Qwen-style, `overlapped` for DeepSeek V4-style), and
   builds `RuntimeConfig` (`_build_runtime_config`), whose KV-cache group
   layout comes from functions the family's runner module exports.
4. `EngineConfig.executor_cls` is a string; `ServingWorker._resolve_executor_cls`
   (`pypto_serving/serving/server/serving_worker.py`) maps it to the imported
   executor class. New families add one branch there.
5. The executor (e.g. `Qwen314BPyptoExecutor`) subclasses `CorePyptoExecutor`
   from `pypto_serving/model/common/executor/`, loads kernel modules from the
   `pypto-lib` submodule (or `$PYPTO_LIB_ROOT`), and stages weights following
   the family's `weight_spec.py`.
6. The runner (e.g. `Qwen314BModelRunner`, an `L3DispatchMixin` + `ModelRunner`
   subclass) issues decode/prefill via the shared L3 dispatch machinery in
   `pypto_serving/model/common/runner/`.

Keep this chain in mind for orientation: which points need a new branch
depends on the triage path below. A same-family adaptation (Path A) reuses
the executor and runner, but may extend family recognition, loader selection,
or weight rules. A new family adds roughly one branch at each point, plus
the weight spec that feeds the executor.

## Step 1 — Triage

Read the checkpoint before writing code: `config.json` (`model_type`,
`architectures`, layer/expert counts, dtype fields), the tokenizer files
present, and the weight shard layout (`model.safetensors.index.json` or a
single shard). Then classify:

| Path | When | Scope | Reference |
| --- | --- | --- | --- |
| A. Same family, new checkpoint or quant | `config.json` positively identifies an existing family (not via the Qwen fallback) **and** the checkpoint matches the executor's kernel-baked shapes and quantization format (see below) | Loader recognizer, weight rules, config validation | `DeepSeekV4W8A8DirectoryLoader` |
| B. New dense architecture | Decoder-only, no MoE | New `model/<family>/` module for a new architecture; smaller contract deltas extend the existing module instead | `model/qwen/` |
| C. New MoE architecture | Expert routing in config | New module plus expert sharding and cache partitions for a new architecture; smaller deltas extend the existing module instead | `model/deepseek/`, `model/deepseek_dspark/` |

A family match alone does not make Path A. Executors bake kernel contracts
for one specific model: the Qwen executor's `_validate_supported_shape`
(`pypto_serving/model/qwen/npu_executor.py`) hard-codes the Qwen3-14B layer
shapes — hidden 5120, intermediate 17408, attention heads 40/8, head_dim 128,
plus page size — and rejects anything else at load. Quantization is the other
common disqualifier: the Qwen staging rules cast projection weights straight
to bf16 (`LayerWeightRule(..., torch.bfloat16, transpose=True)` in
`qwen/weight_spec.py`) because the kernels behind them are bf16 kernels. A
same-shape W8A8 checkpoint carries int8 weights plus scales that those rules
would silently miscast — it needs a quantization-aware loader and quantized
kernels (the `DeepSeekV4W8A8DirectoryLoader` + w8a8 path). Before
classifying as Path A, diff the checkpoint against that baked contract item
by item: layer shapes, layer count, dtype, quantization format, KV layout. A
same-family checkpoint with different shapes or an incompatible quantization
format leaves Path A — route it by architecture and required work: dense
models continue as Path B, MoE models as Path C — and size the scope by the
actual contract delta: a same-family mismatch usually extends the existing
module (kernel shapes, an executor or runner variant, weight rules) rather
than creating a new `model/<family>/` module, which is reserved for genuinely
new architectures.

Effort differs by roughly an order of magnitude across paths — say which path
you took and why in the final report before implementing.

Inventory to collect and confirm with the user when ambiguous:

- dtype and quantization: bf16? w8a8? pre-quantized checkpoint or does it need
  an offline conversion script (see `scripts/convert_deepseek_v4_to_w8a8.py`)?
- layer count, hidden size, head layout; for MoE: expert count, routing shape,
  shared experts, compress ratios if the family uses them.
- KV cache layout the kernels expect (grouped windows? MLA-style compression?).
- tokenizer: standard HF tokenizer files loadable through
  `pypto_serving/model/tokenizer.py`'s `TokenizerAdapter`?
- weight layout: safetensors shards plus a weight map. Multiple shards without
  `model.safetensors.index.json` are rejected by `_load_safetensors_weight_map`
  — fix the checkpoint or generate the index before anything else.
- topology: card count, replica vs overlapped placement, expert parallel size.

## Step 2 — Family Detection and CLI Wiring

Each extension below has its own trigger; apply only the ones the chosen path
sets off — Path A may need a recognizer update but usually reuses the CLI and
executor wiring. An executor variant within an existing family does not need
a new family literal.

- **New family or unrecognized checkpoint variant** — in
  `pypto_serving/model/model_family.py`, extend the `ModelFamily` literal only
  for a new family; add or update a recognizer when an existing family's new
  checkpoint is not positively identified. Check `model_type` and lowercased
  `architectures` entries. **Do not ride the fallback**: `detect_model_family`
  silently returns `"qwen"` for unknown configs, so that return value alone
  does not prove a Qwen match. If you cannot detect a family with confidence,
  make detection raise.
- **Family behavior, executor variant, or supported topology changes** —
  `pypto_serving/cli/main.py`, `build_serving_engine_config`: add branches
  for executor kwargs (e.g. `compile_kernels`), speculative-decoding support
  and its validation, `ParallelConfig` (dp/tp/ep sizes, `placement_mode`),
  and `_validate_model_topology` (reject unsupported card counts loudly
  instead of letting layout assertion fail on device — a reused executor on a
  new topology still needs this validation updated).
- **New executor (family or variant)** —
  `_executor_cls_for_model_family`: return a new executor id string
  (`Pypto<Family>Executor`).
- **Whenever a new executor id appears** —
  `pypto_serving/serving/server/serving_worker.py`,
  `_resolve_executor_cls`: map that string to the imported class. Note
  `pypto_serving/serving/engine/async_engine.py` keeps a default executor id —
  leave the default alone and always pass the family explicitly.
- **When the family's kernel contract changes** — `_build_runtime_config`:
  this is where kernel contracts surface as config —
  KV-cache group specs, `max_prefill_tokens_per_request`, prefill chunk
  choices, homogeneous prefill/decode requirements. The DeepSeek branches
  import these from the family's runner/executor modules
  (`build_deepseek_v4_cache_group_specs`,
  `load_deepseek_v4_serving_contract`); follow that pattern so kernel limits
  stay defined beside the kernels, not duplicated in the CLI.

## Step 3 — Loader and Tokenizer

- `pypto_serving/model/model_loader.py` holds a registry, `ModelLoader`,
  defaulting to `[DeepSeekV4W8A8DirectoryLoader(), HuggingFaceDirectoryLoader()]`
  — first match wins, so family-specific loaders must register before the
  generic `HuggingFaceDirectoryLoader`.
- A loader implements the `ModelFormatLoader` protocol (`format_names`,
  `supports_format`, `can_load`, `load`) and usually derives from
  `SafetensorsDirectoryLoader`, which reduces `can_load` to "config.json
  exists" plus a family-specific `_recognises` hook.
- Path A quantizations: follow `DeepSeekV4W8A8DirectoryLoader` — it validates
  the quantized checkpoint contract (`_validate_deepseek_v4_weight_index`)
  before load, so bad checkpoints fail on CPU with named missing tensors.
- `_build_model_config` and `_build_layer_specs` translate `config.json` into
  the runtime model description. Extend them for new config fields; fork them
  only for a genuinely different model description.
- Tokenizer: confirm the checkpoint loads through `TokenizerAdapter` early —
  executor debugging with an unvalidated tokenizer wastes device time.

## Step 4 — Weight Staging Spec

Kernels consume stacked whole-model slabs, not per-tensor HF weights. The
family's `model/<family>/weight_spec.py` declares the mapping using the rule
DSL in `pypto_serving/model/common/weights/spec.py`:

- `LayerWeightRule` — one kernel weight per layer, cast to the kernel dtype
  (projections bf16, norm gammas fp32), with `transpose=True` when the
  checkpoint ships `[out, in]` and the kernel wants `[in, out]`, and
  `flatten_to_row` for 1-D gammas so they stack as `[1, dim]` rows.
- `DefaultedWeightRule` — absent checkpoint tensor with a neutral default:
  Qwen3's optional `q_norm`/`k_norm` default to **ones**, not zeros — zeros
  annihilate activations; an absent tensor can be a variant, not a fault.
- `OptionalWeightRule` / `SyntheticWeightRule` / `ExpertWeightRule` — skip,
  synthesize, or expert-shard a tensor (see `deepseek/weight_spec.py` for the
  MoE forms).
- `GlobalWeightRule` — whole-model tensors. Padding semantics matter: the
  embedding pads with zeros to the kernel's vocab multiple (Qwen:
  `QWEN_VOCAB_PAD_MULTIPLE = 512`), but an LM head pads with `first_row` — an
  LM head cannot pad with zeros. A tied checkpoint falls back to the embedding
  via `fallback_source`.
- Rule order is contract: it defines slab layout. Do not reorder rules without
  checking the kernel's expected layout.
- Rank axes differ by family: Qwen stacks layers on axis 0 with no rank axis
  (`NoShard` policy — one copy per weight); DeepSeek weights lead with the
  rank axis and stack on axis 1 because upload shards distribute ranks. Pick
  the axis convention from the kernels before writing rules.
- `StackGroup` groups layers into slabs; `StagingPolicy` controls parallel
  staging (workers per layer, `pin_torch_threads=True` — without pinning,
  staging threads fan out into torch's pool and oversubscribe the host).
- NZ-format repacking for fused kernels lives in
  `pypto_serving/model/common/weights/nz.py` and `packer.py`; offline
  conversions go in `scripts/` (`convert_deepseek_v4_to_w8a8.py` is the
  reference).

Validate the spec on CPU before device time, with two distinct checks:

- **Staging correctness** — resolve the rules and assert the staged layout
  (dtypes, transposes, stacking) matches what the executor stages;
  `tests/unit/model/qwen/test_qwen_weight_staging.py` is the pattern. Note
  that its checkpoints are synthesized from known weights, so it proves
  staging behavior, not real-checkpoint coverage.
- **Real-checkpoint coverage** — diff the *actual* checkpoint's
  `model.safetensors.index.json` (or single-shard tensor list) against the
  rules in both directions, judging each rule's source requirement by that
  rule's actual resolution conditions:
  `LayerWeightRule` / `ExpertWeightRule` sources must be present;
  `OptionalWeightRule` sources are expected exactly when
  `enabled_for(compress_ratio)` says the layer kind carries them;
  `DefaultedWeightRule` sources may be absent as `required_when` permits;
  `GlobalWeightRule` sources may be absent when their `fallback_source` is
  present (a tied checkpoint ships no `lm_head.weight`); `SyntheticWeightRule`
  sources are expected nowhere. In the other direction, every index entry
  must be consumed by a rule or sit on an explicit ignore list. Extra
  unconsumed tensors pass the synthesized tests silently and surface later as
  mis-loaded weights — enumerate the ignore list in the PR so reviewers can
  challenge it.

Layout mismatches are cheap on CPU and expensive as SMMU faults on device.

## Step 5 — Executor and Runner

Clone the reference family as the skeleton
(`npu_executor.py`, `npu_runner.py`, `task_args.py`, `weight_spec.py`):

- Executor: subclass `CorePyptoExecutor`
  (`pypto_serving/model/common/executor/pypto_executor.py`); sampler and
  utilities are shared. Kernel module discovery follows
  `_find_pypto_lib_qwen14b_dir`: `$PYPTO_LIB_ROOT` or the repo's `pypto-lib`
  submodule. When kernel constant names drift across pypto-lib generations,
  prefer a compat shim over a hard pin (see `_kernel_batch_pad` accepting both
  `BATCH` and `BATCH_PAD`).
- Runner: subclass `ModelRunner` with `L3DispatchMixin`
  (`pypto_serving/model/common/runner/`); per-family layout, compiled-kernel
  cache, prefill/decode input assembly, and static kernel args
  (`task_args.py`). Shared buffer management is `buffer_set.py` — reuse it.
- The L3 worker forks lazily at the first device-side call, and host-shared
  buffers must be allocated *before* that fork or they SMMU-fault (fault
  signature in `references/device-fault-triage.md`, row 3). Keep the runner's
  `_ensure_l3_shared_buffers` ordering: host-shared staging allocations first,
  then weight upload, then device scratch — never allocate host-shared buffers
  at dispatch time.
- Export the family's cache-group and serving-contract builders from the
  runner/executor modules so `_build_runtime_config` can import them (Step 2).

## Step 6 — Kernels in pypto-lib

- Survey `pypto-lib/` for ops the new model needs before writing any: missing
  ops are the usual long pole for a new architecture, and finding them during
  Step 4 instead of Step 5 changes the plan.
- Kernel layouts can be frozen from CLI flags (DSpark freezes import
  arguments, cache partitions, and packed-prefill capacity from `--ep`); if
  the new model parameterizes kernels, decide which flags freeze which
  layouts and validate them in `_validate_model_topology`.
- Kernel-source changes belong in the `pypto-lib` repo at its pinned
  submodule commit; bumping the pin is a separate review from the serving-side
  integration.

## Step 7 — Tests, Docs, Examples

- Unit tests: `tests/unit/model/<family>/` — mirror the reference family's
  coverage (`test_npu_executor.py`, `test_npu_runner_inputs.py`,
  `test_qwen_weight_staging.py`-style spec coverage). Shared-machinery tests
  live in `tests/unit/model/common/` and `tests/unit/cli/`
  (`test_parallel_options.py` for new topology flags).
- Docs: a user-guide page under `docs/user-guide/` with the offline and HTTP
  example commands; update `docs/developer-guide/model-integration.md` only
  if the integration checklist itself changed.
- Lint and tests must pass before any device run:
  `python -m pytest tests/unit`, `python tests/lint/check_headers.py`,
  `python tests/lint/check_english_only.py`,
  `ruff check --config ruff.toml .`. New `.py` files need the repository
  copyright header (see any existing file).

## Device Validation

Stage validation cheapest-first; each stage gates the next:

1. CPU: unit tests plus the weight-spec coverage check from Step 4.
2. First device smoke: `task-submit` per `AGENTS.md`, small bounds
   (`--max-model-len 512`, `max_new_tokens` around 5, one prompt), with logs
   and timeouts pre-configured (template below).
3. Full-length offline generation: several prompts, contract assertions only
   (fault table row 5) — a legal finish reason (`stop`/EOS, or `length` when
   `max_new_tokens` caps the run), non-empty output, token accounting closes.
4. HTTP path: start the server, issue a completion, confirm the served model
   name and a coherent response.
5. Optional perf smoke: reuse the `qwen3-14b-offline-perf-test` skill's
   reporting format so numbers stay comparable across models.

Environment template for every device run — logs land in the worktree, first
dispatch survives its multi-minute warm-up, and the topology is explicit:

```bash
ASCEND_PROCESS_LOG_PATH=<worktree>/ascend-logs \
SIMPLER_OP_EXECUTE_TIMEOUT_US=400000000 \
SIMPLER_STREAM_SYNC_TIMEOUT_MS=440000 \
SIMPLER_SCHEDULER_TIMEOUT_MS=320000 \
task-submit --device <device-spec> --max-time 0 --run \
  "PTO2_RING_HEAP=536870912 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 \
  pypto-serving --model <model-dir> --prompt '...' --platform <p> \
    --devices <id,...> --dp <dp> --tp <tp> --ep <ep> \
    --max-model-len 512 --generate-config '{\"max_new_tokens\": 5}'"
```

Topology rules for filling the placeholders:

- `task-submit --device auto` allocates a **single card** — valid only for
  single-card models. Multi-card families need a device spec covering the
  topology's card count.
- The serving command must carry the `--devices`, `--dp`, `--tp`, `--ep`
  values that `_validate_model_topology` accepted. DeepSeek V4, for example,
  requires exactly eight devices with `--dp 8 --ep 8 --tp 1` (see
  `docs/user-guide/deepseek-v4.md`); dspark takes its rank count from `--ep`.
- Record the allocated device ids and parallel sizes in the report — a run
  on the wrong topology is not a validation of the model.

(Preserve the ring-sizing variables the `AGENTS.md` example uses — or the
`--ring-*` CLI flags the family's user-guide page specifies; add the
`SIMPLER_*` timeouts for any multi-card first dispatch.)

Verify the `ascend-logs` directory actually received files before trusting a
"no fault in logs" conclusion.

On any device failure, classify against
`references/device-fault-triage.md` **before** reading model code — most
first-attempt failures are rows 1, 3, or 7, none of which are model bugs.

Record in the final report: repo commit, stack provenance (pypto / simpler /
ptoas), model path, card count and topology, command line, generated samples,
and the contract-check results.

## Landing the Change

- Open a pull request against `main` following the repository's contribution
  conventions; scope it to the model integration only.
- Keep the serving-side integration and any `pypto-lib` pin bump as separate
  PRs; the pin bump lands first, or the integration PR carries the pin change
  with a justification.

## Definition of Done

- [ ] Env stack refreshed/verified against the intended pins; provenance
      recorded.
- [ ] The checkpoint is positively identified and routed to the correct
      family from `config.json`, never via the Qwen fallback; add a recognizer
      for a reused family when the existing one does not match.
- [ ] Loader recognizes the checkpoint and rejects corrupt ones on CPU with
      named missing tensors.
- [ ] Weight spec staging behavior unit-tested, and real-checkpoint index
      coverage diffed both directions against the rules with an explicit
      ignore list.
- [ ] Unit tests and the three lint checks pass.
- [ ] Offline generation runs on device within small bounds, then at full
      length, producing output that meets the contract.
- [ ] The HTTP serving path starts and answers with the new model.
- [ ] Topology and placement mode are explicit in the CLI and validated, not
      inferred.
- [ ] `docs/user-guide/` page and example commands added.
- [ ] Triage path (A/B/C) and final topology recorded in the report.
