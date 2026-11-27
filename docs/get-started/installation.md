# Installation

Install PyPTO Serving from a source checkout. The package does not vendor the Ascend runtime, PyPTO runtime, PyTorch, or model weights; those must be provided by the active environment.

## Before You Start

Prepare a Linux host with Ascend NPUs visible to the serving user, an Ascend-compatible Python 3.10+ environment, CANN, PyPTO runtime pieces, PyTorch, and the model checkpoint you plan to run.

Checkpoint format and device requirements depend on the model. See the [Qwen3-14B](../models/qwen3-14b.md#checkpoint) and [DeepSeek V4](../models/deepseek-v4/index.md) guides for the currently supported paths.

For an environment matching CI, use Python 3.10: the PTOAS wheel selected by
CI targets CPython 3.10. Other Python versions need a compatible PTOAS build.

## Select the Runtime Build Target

When installing PyPTO's `runtime/` submodule (simpler), select the NPU platform
that serving will use. For A2/A3, after activating the matching CANN environment
and installing the Torch build selected for this stack, install the build
prerequisites and PyPTO before simpler:

```bash
python -m pip install 'scikit-build-core>=0.10' 'nanobind>=2,<3' 'cmake>=3.20' ninja
python -m pip install --no-build-isolation /path/to/pypto
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

## Prepare a Project Environment

Create an isolated environment in the serving checkout. If Python 3.10 is
provided by another environment manager, use it to create the `.venv` prefix
instead. Do not enable system site packages: they can mask a missing or stale
PyPTO installation.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

The `pypto-lib` submodule provides kernel sources, not an installed PyPTO
compiler or simpler runtime. For a CI-matched stack, select the target platform
in `.github/workflows/ci.yml` and use that job's `pypto-repository`
and `pypto-revision` to check out PyPTO separately. Initialize its submodules;
the `runtime/` submodule pins simpler. Read PyPTO's
`toolchain/versions.env` for the PTOAS release and architecture-specific
SHA256, and `runtime/pto_isa.pin` for PTO ISA. The
`.github/actions/setup-ci-job/action.yml` documents the
corresponding Torch wheel source, PTOAS download/checksum, and build steps.
Install Torch before PyPTO so dependency resolution does not select a different
build from the default index. Install PyPTO and its pinned `runtime/` into
`.venv`, using the platform-specific build target described above. Install the
verified PTOAS wheel in `.venv` or an isolated environment, and set
`PTOAS_ROOT` to the directory containing `bin/ptoas`. For an isolated PTOAS
environment, prepend its site-packages to `PYTHONPATH` so serving's Python
finds that wheel ahead of any CANN-bundled copy, as the CI setup action does.

## Install the Python Package

Install the runtime requirements and the package in editable mode in the
active environment:

```bash
python -m pip install pytest -r pypto_serving/requirements-runtime.txt
python -m pip install --no-deps -e .
```

`--no-deps` is intentional: PyTorch, PyPTO, simpler, and serving's runtime
requirements must already be installed in the active environment.

## Verify the Install

After installation, confirm the CLI and the installed runtime paths:

```bash
python -c 'import pypto, _task_interface; print(pypto.__file__, _task_interface.__build_commit__)'
if [ -x "$PTOAS_ROOT/bin/python" ]; then
  "$PTOAS_ROOT/bin/python" -c 'import ptoas; print(ptoas.__file__)'
else
  python -c 'import ptoas; print(ptoas.__file__)'
fi
"$PTOAS_ROOT/bin/ptoas" --version
python -c 'from simpler_setup.pto_isa import ensure_pto_isa_root; print(ensure_pto_isa_root())'
pypto-serving --help
```

Check that the simpler build commit matches the selected PyPTO `runtime/`
submodule and the resolved PTO ISA checkout matches `runtime/pto_isa.pin`.
If CANN adds its own packages to `PYTHONPATH`, make sure imports resolve to
the installed environment, not CANN's bundled copy.

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
