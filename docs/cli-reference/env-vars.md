# Environment Variables

Most PyPTO Serving configuration should be passed as CLI arguments. Environment variables are reserved for settings that must be visible before CLI parsing or inside worker subprocesses, such as library discovery, build directories, profiling defaults, worker timeouts, and low-level runtime controls.

## Public Variables

| Variable | Used by | Purpose |
| --- | --- | --- |
| `PYPTO_LIB_ROOT` | Model executors | Path to a `pypto-lib` checkout when it cannot be discovered from an editable source tree. |
| `PYPTO_PROG_BUILD_DIR` | Worker startup | Directory for generated kernel artifacts and compile-cache subdirectories. |
| `PYPTO_STAGING_THREADS` | Qwen weight staging | Optional thread count for Qwen parallel weight staging. |
| `PYPTO_WORKER_INIT_TIMEOUT` | Engine startup | Worker initialization timeout in seconds for large checkpoints or slow first launches. |
| `SERVING_WORKER_STEP_TIMEOUT` | Engine runtime | Worker step timeout in seconds. |
| `PYPTO_SERVING_GC_DEBUG` | Serving utilities | Set to `1` to log garbage-collection debug events in the process. |
| `SA_PROFILE_OUTPUT` | Profiling library and merge script | Profile output path for library users or `scripts/merge_profile.sh`. CLI users should prefer `--profile-output`. |
| `SA_PROFILE_LEVEL` | Profiling library | Comma-separated profile levels. CLI users should prefer `--profile-level`. |
| `PYPTO_RUNTIME_LOG` | PyPTO runtime | Runtime log control forwarded through the process environment. |

## Compile Cache

Set `PYPTO_PROG_BUILD_DIR` to a persistent directory and start with `--use-compile-cache` when later launches should reuse compiled kernels:

```bash
PYPTO_PROG_BUILD_DIR=/data/cache/pypto-build \
pypto-serving \
  --model /path/to/Qwen3-14B \
  --platform a2a3 \
  --device 0 \
  --use-compile-cache \
  --prompt 'Huawei is' \
  --generate-config '{"max_new_tokens":5}'
```

The compile cache has no fingerprint validation. Reuse it only with the same model configuration, assigned devices, kernel sources, ring sizing, and runtime flags. Clear the directory after changing any of those inputs.

## PyPTO Library Discovery

Editable checkouts discover the bundled `pypto-lib/` submodule automatically.

```bash
PYPTO_LIB_ROOT=/path/to/pypto-lib pypto-serving --model /path/to/Qwen3-14B
```

If model startup reports that a kernel directory is missing, check that this variable points to the checkout containing the required model kernel directory. Leave it unset in editable repository checkouts unless automatic discovery fails.

## Profiling Environment

HTTP serving should use explicit CLI options:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --profile \
  --profile-output /tmp/pypto-profile \
  --profile-level e2e,kernel
```

`SA_PROFILE_OUTPUT` and `SA_PROFILE_LEVEL` remain useful for lower-level library runs and for profile merge helpers:

```bash
SA_PROFILE_OUTPUT=/tmp/pypto-profile ./scripts/merge_profile.sh
```

Do not set `SA_PROFILE_MAIN_PID` manually. It is an internal coordination value created by the profile startup code.

## Worker Timeouts

Large checkpoints and cold cache launches can take longer than the default worker initialization timeout. Raise `PYPTO_WORKER_INIT_TIMEOUT` when model loading, weight staging, or first compilation legitimately needs more time:

```bash
PYPTO_WORKER_INIT_TIMEOUT=1800 pypto-serving --model /path/to/dsv4-flash-w8a8
```

`SERVING_WORKER_STEP_TIMEOUT` should only be raised after confirming that device execution is still progressing. A step timeout usually means the worker or NPU dispatch is hung, so increasing it can hide a real fault.

## Legacy Ring Variables

Older examples may set `PTO2_RING_DEP_POOL`, `PTO2_RING_TASK_WINDOW`, or `PTO2_RING_HEAP`. New serving paths should use `--ring-dep-pool`, `--ring-task-window`, and `--ring-heap` instead. These CLI options are carried per dispatch through `RunConfig` rather than applied as process-wide environment variables.
