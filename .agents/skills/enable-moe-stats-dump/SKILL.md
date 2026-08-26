---
name: enable-moe-stats-dump
description: Enable, disable, and verify DeepSeek V4 MoE physical-expert load statistics in pypto-serving online or offline inference. Use when a user asks to turn on MoE statistics, dump activated expert counts, inspect per-expert routed-token counts, validate the JSONL output, toggle collection without changing the pypto-lib ABI, or troubleshoot an empty or incompatible MoE statistics run.
---

# Enable MoE Statistics Dump

Use the existing fixed pypto-lib ABI and runtime flag. Do not create a separate kernel implementation or
compile-time variant for statistics-on mode.

## Enable collection

Start from a known-working DeepSeek V4 command and append an explicit output path:

```bash
--moe-stats-output /absolute/path/to/moe-stats.jsonl
```

The unified CLI accepts the option in both modes:

- Online server: `python -m pypto_serving.cli.main ...`
- Offline generation: `python -m pypto_serving.cli.main ... --prompt TEXT`

Prefer an absolute path. The writer appends JSONL records, so move or remove the exact old output file before
a clean measurement. Never delete an unresolved variable, glob, directory, or broad path.

Do not add the option for Qwen or another model family. The CLI intentionally restricts it to DeepSeek V4.

## Disable or redirect collection

Omit `--moe-stats-output` at startup to disable collection. The disabled path keeps the same compiled callable
ABI but skips statistics-buffer clearing, device-to-host transfer, and JSONL writes.

When controlling an already-created executor in embedded code, call its public method:

```python
executor.set_moe_stats_output("/absolute/path/to/new.jsonl")
executor.set_moe_stats_output(None)
```

Use the executor owned by the application's control plane. Do not reach into private runner dictionaries.

## Generate evidence

Send at least one request that performs prefill and emits at least one decode token. Check all of the following:

1. The request completed with a non-empty output and a positive completion-token count.
2. Enabled mode created a non-empty JSONL file.
3. Disabled mode did not create a new JSONL file. Ensure no stale file existed before making this assertion.
4. The JSONL contains the expected phases and internally consistent expert counts.

Validate and summarize the file with the bundled script:

```bash
python .agents/skills/enable-moe-stats-dump/scripts/inspect_moe_stats.py \
  /absolute/path/to/moe-stats.jsonl \
  --require-phase prefill \
  --require-phase decode
```

For real Ascend execution, follow the environment's NPU scheduling and allocation procedure. Keep model paths,
device IDs, Python environments, and PTOAS locations outside this reusable skill.

## Interpret the JSONL

Each line describes one completed model dispatch:

- `timestamp_ns`: host timestamp for the record.
- `dispatch_id`: writer-local monotonically increasing record ID.
- `phase`: `prefill`, `decode`, `decode_mtp`, `mtp_prefill`, or `mtp_decode`.
- `ranks` and `local_experts`: expert-parallel layout metadata.
- `layers`: one entry per hidden-layer row plus the MTP row.

Each layer contains:

- `layer_id`: zero-based row ID.
- `kind`: `main` or `mtp`.
- `active_experts`: number of physical experts whose count is nonzero.
- `routed_tokens`: sum of route assignments accepted by all physical experts in the layer.
- `expert_token_counts`: rank-major global expert counts. Global expert ID is
  `rank * local_experts + local_expert_id`.

Treat `routed_tokens` as route assignments, not unique input-token count. Top-k routing can make it larger than
the number of model input rows. Main-only phases retain the fixed MTP row, which may be all zeros.

## Troubleshoot

- If the file is absent or empty, first prove that a request actually reached model execution and emitted a
  token. An HTTP 200 response with zero completion tokens is not successful evidence.
- If the CLI rejects the option, confirm the selected model family is DeepSeek V4 and the current checkout
  contains `--moe-stats-output`.
- If runtime reports an old parameter count, such as expecting 84 arguments while receiving 86, suspect a
  stale top-level compile cache from before the fixed statistics ABI. Rebuild or isolate only the relevant
  cache; do not broadly delete unrelated build artifacts.
- If PTOAS reports an unknown operation, select a PTOAS build compatible with the active Python and current
  PyPTO/PTO-ISA toolchain. Verify the version inside the submitted environment rather than assuming the host
  shell and task use the same binary.
- If enabled and disabled generations differ, report both responses, exact runtime versions, and logs before
  attributing the difference to statistics collection.

## Report results

Report the model, entry point, enabled/disabled mode, output path, request completion-token count, record count,
phases, layer count, expert count per layer, and whether routed-token counts are nonzero. Distinguish local/JIT
validation from completed real-NPU validation.
