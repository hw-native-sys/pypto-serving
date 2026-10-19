# Installation

Install PyPTO Serving from a source checkout. The package does not vendor the Ascend runtime, PyPTO runtime, PyTorch, or model weights; those must be provided by the active environment.

## Before You Start

Prepare a Linux host with Ascend NPUs visible to the serving user, an Ascend-compatible Python 3.10+ environment, CANN, PyPTO runtime pieces, PyTorch, and the model checkpoint you plan to run.

Checkpoint format and device requirements depend on the model. See the [Qwen3-14B](../models/qwen3-14b.md#checkpoint) and [DeepSeek V4](../models/deepseek-v4/index.md) guides for the currently supported paths.

## Clone

Clone the repository and initialize the kernel submodule:

```bash
git clone https://github.com/hw-native-sys/pypto-serving.git
cd pypto-serving
git submodule update --init --recursive
```

## Install the Python Package

Install the package in editable mode:

```bash
python -m pip install --no-deps -e .
```

`--no-deps` is intentional. The project expects the Python environment to already contain the Ascend-compatible PyTorch build, PyPTO runtime pieces, and serving dependencies that match the target machine.

For HTTP serving, make sure the environment also has:

```bash
python -m pip install fastapi uvicorn pydantic
```

For model conversion and checkpoint loading, make sure `safetensors` and `transformers` are available:

```bash
python -m pip install safetensors transformers
```

## Required PyPTO API

Build PyPTO from a revision containing [PyPTO #2975](https://github.com/hw-native-sys/pypto/pull/2975), and build `simpler` from that checkout's `runtime` submodule. Merge that PyPTO change before deploying this serving revision. The `pypto` and `simpler` packages on PyPI are unrelated projects and must not be used to satisfy this requirement.

The required API is `RunConfig(backend_type=..., execution_mode=...)`, with both values honored. Some older builds accept `backend_type` but silently keep the default Ascend910B backend, so an import check alone is insufficient. After installing PyPTO, run this check in the same Python environment used for serving; CI runs the same check before model tests:

```bash
python - <<'PY'
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

for mode, platform in ((ExecutionMode.SIM, "a5sim"), (ExecutionMode.ONBOARD, "a5")):
    config = RunConfig(backend_type=BackendType.Ascend950, execution_mode=mode)
    if (config.backend_type != BackendType.Ascend950
            or config.execution_mode != mode or config.platform != platform):
        raise SystemExit("Rebuild PyPTO with hw-native-sys/pypto#2975; target selection is incompatible")
print("PyPTO target API OK")
PY
```

If the check fails, rebuild and reinstall PyPTO and `simpler` from the compatible checkout before starting a server.

## Verify the Install

After installation, confirm the CLI is available:

```bash
pypto-serving --help
```

Then run the Qwen quickstart once the model checkpoint is available:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 5}'
```
