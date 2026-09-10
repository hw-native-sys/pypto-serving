---
name: profile-dsv4-aligned-gbs32-strace
description: Run and analyze the fixed DeepSeek V4 PyPTO alignment benchmark with GBS32, DP8, per-rank batch4, 64 prompt tokens, 256 output tokens, MTP k=1, temperature=0, verbose serving profiling, and eight Simpler Host STRACE lanes. Use when profiling DSV4 serving, comparing PyPTO with the recipe baseline, measuring host overhead, inspecting bind/runner/validate, generating a Perfetto swimlane, or discussing the standard 32-batch 对齐测试.
---

# Profile DSV4 Aligned GBS32

Run the standard alignment workload and generate one combined serving/Host Perfetto trace.
Keep the workload fixed unless the user explicitly requests a separate experiment.

## Fixed workload

- GBS: 32
- parallelism: DP=8, EP=8, TP=1; per-rank batch=4
- prompts: 32 distinct domestic-attraction chat prompts in the fixed index order from
  [assets/prompts.json](assets/prompts.json), exactly 64 model tokens each
- output: 256 tokens per request, `ignore_eos=1`
- sampling: `temperature=0`, `top_p=1`, `top_k=0`
- MTP: enabled, `k=1`, fused one-L2 decode
- serving: chunked prefill enabled, prefix cache disabled
- profiling: verbose serving profiler plus Simpler Host STRACE
- device diagnostics: Device STRACE and device log disabled

The runner performs one unprofiled GBS32 warmup to initialize and compile the same paths.
Only the second GBS32 batch is inside the serving profile window.
Both batches use the same ordered set of 32 distinct inputs, transcribed from the
user-provided alignment set in `/data/jinzongquan/file/prompt.md`; that external file
is not required at runtime. Render each raw prompt exactly as
`<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜></think>`.
Do not truncate, pad, shuffle, or repeat prompts. The compact JSON token-ID matrix
SHA256 must be `346e69fa38fc12f9412564288a16b8a674a47365fa8488c690ff988c9c3fdf95`.

## Acquire devices

Obtain exactly eight devices through the environment's normal scheduler. Do not kill or
reuse devices owned by another task. Run the skill inside the allocation and pass the
assigned device IDs explicitly.

## Run

From the pypto-serving repository root:

```bash
bash .agents/skills/profile-dsv4-aligned-gbs32-strace/scripts/run_profile.sh \
  --model-dir /path/to/dsv4-flash-w8a8 \
  --devices 0,2,4,6,8,10,12,14 \
  --use-compile-cache \
  --artifact-dir /path/to/artifacts \
  --run-id <task-id>
```

The wrapper also accepts `PYPTO_DSV4_MODEL_DIR`, `PYPTO_PROFILE_PYTHON`,
`PYPTO_PROFILE_DEVICES`, `PYPTO_PROFILE_RUN_ID`, and `PYPTO_USE_COMPILE_CACHE=1`.
Set `PYPTO_RUNTIME_ROOT` when `simpler_setup` is not installed in the selected Python
environment.

Reuse compile cache only with the same devices, commits, model configuration, and kernel
sources. The cache does not reliably reject stale binaries.

## Trace contract

The skill produces:

- `serving-trace/trace.json`: real scheduler, serving, worker, executor, and kernel spans
- `simpler-swimlane.json`: Host-only Simpler invocations overlapping the formal profile
- `serving-strace-swimlane.json`: the serving spans plus eight Host STRACE tracks on the
  shared host monotonic clock
- `profile-summary.{json,md}`: workload, performance, callable classification, and the
  observed fused decode Steps
- `skill-profile-validation.json`: final artifact validation
- `prompt_manifest.json`: ordered raw/rendered inputs, attractions, token IDs, and per-input hashes
- `responses.json`, `performance_summary.json`, `server.log`, and `run.log`: raw evidence

Open `serving-strace-swimlane.json` in Perfetto. Use its `scheduler/worker/executor/kernel`
tracks for framework flow and the eight `strace.host` tracks for
`bind/runner_run/validate` attribution.

Device STRACE is intentionally disabled because it perturbs timing. Therefore device wall,
orchestrator, scheduler-device, and Effective timing are unavailable in this skill run;
never report them as zero or infer them from Host Step.

## Validate

Require all of the following before reporting success:

1. The runner exits with code 0 and writes 32 responses of exactly 256 tokens.
2. All 32 rendered prompts and token sequences are unique, each has exactly 64 tokens,
   and their per-input hashes and ordered matrix SHA256 match the locked alignment set.
3. `serving-trace/trace.json` contains `scheduler`, `serving`, `worker`, `executor`, and
   `kernel` spans.
4. `server.log` contains Host STRACE and no `clk=dev` record.
5. The combined trace contains exactly eight Host device tracks.
6. The combined trace contains the same continuous fused decode Step sequence on every rank.
7. `skill-profile-validation.json` reports `valid: true`.

Treat profiler-instrumented TPOT as diagnostic. Use an equivalent unprofiled run for the
official performance number.

## Reprocess

Rebuild analysis and the combined trace without devices:

The artifact must include `prompt_manifest.json` from the 32-distinct-input runner.
Old repeated-prompt artifacts do not satisfy this workload and must be rerun.

```bash
python .agents/skills/profile-dsv4-aligned-gbs32-strace/scripts/analyze_profile.py \
  <artifact-dir> --run-id <task-id>

python .agents/skills/profile-dsv4-aligned-gbs32-strace/scripts/render_8lane.py \
  <artifact-dir>/simpler-swimlane.json \
  <artifact-dir>/server.log \
  <artifact-dir>/serving-strace-swimlane.json \
  --serving-trace <artifact-dir>/serving-trace/trace.json \
  --profile-summary <artifact-dir>/profile-summary.json

python .agents/skills/profile-dsv4-aligned-gbs32-strace/scripts/validate_artifact.py \
  <artifact-dir>
```
