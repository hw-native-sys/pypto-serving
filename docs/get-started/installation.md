# Installation

Install PyPTO Serving from a source checkout. The package does not vendor the Ascend runtime, PyPTO runtime, PyTorch, or model weights; those must be provided by the active environment.

## Before You Start

Prepare a Linux host with Ascend NPUs visible to the serving user, an Ascend-compatible Python 3.10+ environment, CANN, PyPTO runtime pieces, PyTorch, and the model checkpoint you plan to run.

Checkpoint format and device requirements depend on the model. See the [Qwen3-14B](../models/qwen3-14b.md#checkpoint) and [DeepSeek V4](../models/deepseek-v4/index.md) guides for the currently supported paths.

## Select the Runtime Build Target

When installing PyPTO's `runtime/` submodule (simpler), select the NPU platform
that serving will use. For A2/A3, after activating the matching CANN environment,
install the build prerequisites before using `--no-build-isolation`:

```bash
python -m pip install 'scikit-build-core>=0.10' 'nanobind>=2,<3' 'cmake>=3.20' ninja
python -m pip install --no-build-isolation /path/to/pypto/runtime \
  --config-settings=build.targets=build_package_a2a3
```

Use `build_package_a5` for A5, with a CANN SDK that provides the required HCCL
channel-config APIs. The host CPU architecture (`aarch64` or `x86_64`) does not
select the NPU platform. An unrestricted runtime build can also attempt A5 on
an A2/A3 host and fail because its SDK lacks those APIs.

The CI setup action takes the same choice through `runtime-platform`; the
A2/A3 serving job selects `a2a3`, and the V4.1 A5 job selects `a5`.

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
