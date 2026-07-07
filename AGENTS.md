# Repository Guidelines

## Project Layout

PyPTO Serving is a local inference stack for Qwen3-14B generation on Ascend
NPUs. Runtime code lives under `python/`, with CLI entry points in
`python/cli/` and engine, scheduler, KV cache, model loading, sampling, and
streaming logic in `python/core/`. Qwen3-14B examples and executor glue live
under `examples/model/qwen3_14b/`. The `pypto-lib/` submodule provides the
kernel sources used by the NPU executor. Unit and lint tests live under
`tests/`.

## Skills

Repository-local agent skills are stored canonically in `.agents/skills/`.
The `.claude/skills` and `.codex/skills` paths are symlinks to
`../.agents/skills`, so Claude-compatible and Codex-compatible tooling can use
the same skill definitions without duplicate files. When adding or updating a
skill, edit `.agents/skills/<skill-name>/SKILL.md`.

## Development Commands

Initialize submodules after cloning:

```bash
git submodule update --init --recursive
```

Run the main unit tests:

```bash
python -m pytest tests/test_batching.py tests/test_parallel.py
```

Run lint checks used by pre-commit:

```bash
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
ruff check --config ruff.toml .
```

## Style And CI

Use Python 3.10-compatible code. Ruff is configured in `ruff.toml` with line
length 110, `target-version = "py310"`, `F` checks enabled, and `F841` ignored.
GitHub CI runs pre-commit plus the CLI and batching unit tests. Keep generated
artifacts such as `build_output/`, caches, and compiled files out of commits.

## NPU Verification

Device verification usually runs through `task-submit`. Preserve the runtime
environment variables used by the examples unless the task explicitly changes
them:

```bash
task-submit --device auto --max-time 0 --run \
  "PTO2_RING_HEAP=536870912 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 \
  python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /data/linyifan/models/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5"
```

Use `--profile` for timing reports.
