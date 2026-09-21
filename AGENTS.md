# Repository Guidelines

## Project Layout

PyPTO Serving is a local inference stack for Qwen3-14B and DeepSeek V4
generation on Ascend NPUs. Installable runtime code lives under
`pypto_serving/`, with CLI entry
points in `pypto_serving/cli/`, serving orchestration in
`pypto_serving/serving/`, and model integrations in `pypto_serving/model/`.
Runnable scripts and configuration stay under `examples/`. The `pypto-lib/`
submodule provides kernel sources used by the NPU executors. Unit and lint tests
live under `tests/`.

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
python -m pip install --no-deps -e .
```

Run the main unit tests:

```bash
python -m pytest tests/unit
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
GitHub CI always runs pre-commit static checks. Pull requests and pushes that
only change `.agents/skills/` or the `.claude/skills` and `.codex/skills` aliases
skip the NPU test and platform build jobs. Mixed changes, manual workflow runs,
and failed or missing change classifications still run the full gates unless
the workflow is cancelled. Keep generated artifacts such as `build_output/`,
caches, and compiled files out of commits.

This exemption assumes skill helper scripts are standalone tools that are not
imported or executed by the serving runtime or its test suites. If that boundary
changes, update the CI path classification before relying on those scripts.
Keep the canonical skill path and its aliases above in sync with `skill_roots`
in `.github/workflows/ci.yml`.

## NPU Verification

Device verification usually runs through `task-submit`. Preserve the runtime
environment variables used by the examples unless the task explicitly changes
them:

```bash
task-submit --device auto --max-time 0 --run \
  "PTO2_RING_HEAP=536870912 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 \
  pypto-serving --model /data/linyifan/models/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-model-len 512 \
    --generate-config '{\"max_new_tokens\": 5}'"
```

Use `--profile` for timing reports.
