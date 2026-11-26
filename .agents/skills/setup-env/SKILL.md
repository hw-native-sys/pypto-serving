---
name: setup-env
description: Build or repair pypto-serving's project-local .venv for CPU tests or Ascend serving, using the selected CI platform's compatible PyPTO, simpler, PTO ISA, PTOAS, and Torch stack. Use before local system tests or when the serving environment is missing or mismatched.
---

# Set Up the Serving Environment

Read [Installation](../../../docs/get-started/installation.md) and use
`.github/workflows/ci.yml` plus `.github/actions/setup-ci-job/action.yml` as
the executable reference. Read the matching job (`a2a3` or `a5`), including
its PyPTO repository/revision overrides; do not copy a pin from another
platform or hard-code a pin in this skill. The serving checkout pins its kernel
sources in the `pypto-lib` submodule, not the installed compiler/runtime.

## Prepare

1. Inspect `git status`, submodule state, `.venv`, host Python/toolchain,
   `uname -m`, CANN root, and any existing PyPTO/ISA checkouts before changing
   anything. Select the target from the requested model and device, not CPU
   architecture. Do not update the serving branch or replace an existing
   checkout unless the user requests it; preserve unrelated local changes.
2. Initialize `pypto-lib` at this checkout's pin. Resolve the PyPTO revision
   from the selected CI job, or the user's explicit target. Use a separate,
   user-writable PyPTO checkout; initialize its submodules, including
   `runtime/` (simpler). Read `toolchain/versions.env` for PTOAS version and
   architecture-specific SHA256, and `runtime/pto_isa.pin` for the ISA commit.
3. Create the project-local `.venv` with Python 3.10. CI's bundle and its
   isolated `python -m venv` are preferred when available; a Conda prefix at
   `.venv` is a valid local alternative when the bundle/Python 3.10 is missing.
   Do not enable system site packages: an older host PyPTO can conceal a failed
   install. Reuse a populated `.venv` only after verifying its installed pins;
   never delete an existing environment without inspecting it first.

## Install

- Match the selected CI job's Torch source/version and build prerequisites.
  In particular, use its CPU-wheel index when CI does so; do not let pip
  substitute a CUDA-bearing wheel from the default index. Install PyPTO from
  the selected checkout, then install its pinned `runtime/` submodule using
  `--no-build-isolation` and `build.targets=build_package_<platform>` as in
  CI. Source the matching CANN SDK for device targets before building simpler.
  CPU compilation and installs do not need an NPU allocation.
- Install the PTOAS wheel for the host CPU architecture at the version and
  verified SHA256 from that PyPTO checkout. Keep it in the project environment
  (or an isolated venv beneath `.venv`) and point `PTOAS_ROOT` to the directory
  containing its `bin/ptoas`. Install `pytest`,
  `pypto_serving/requirements-runtime.txt`, and this repo with
  `pip install --no-deps -e .` inside `.venv`.
- Provide a repeatable activation command in the ignored `.venv` directory.
  Activate `.venv`, source the chosen CANN `set_env.sh` for device work, export
  `PTOAS_ROOT`, and put the environment's `bin` and site-packages before CANN's
  own `ptoas`/`pypto` in `PATH` and `PYTHONPATH`. Check the actual import and
  executable paths after activation. Avoid leaking Conda's OpenSSL libraries
  into Git HTTPS subprocesses; if cloning fails with `libldap`/`EVP_md2`, fix
  the library path or use a verified local checkout rather than retrying on a
  device lease.
- `PTO_ISA_ROOT` alone does **not** redirect simpler's managed ISA checkout.
  Before borrowing a card, call `simpler_setup.pto_isa.ensure_pto_isa_root()`;
  it must resolve a clean checkout at `runtime/pto_isa.pin`. If network access
  is unavailable, a clean local checkout at that exact commit may be linked
  into simpler's managed build location. Never overwrite a dirty checkout.

## Verify

Check Python version; `pypto.__file__`, `ptoas.__file__`, `ptoas --version`,
Torch build, and `_task_interface.__build_commit__` against the selected
PyPTO runtime submodule. Verify `ensure_pto_isa_root()` against the ISA pin,
and confirm the requested simpler platform was built. Run the relevant CI CPU
guards before any device test. For an explicitly requested NPU smoke/ST, read
the `task-submit` skill and `task-submit --skill`; only submit the prepared
device workload through the queue, with the user's model path and the CI job's
device count/platform. Cold JIT compilation inside a model test cannot always
be separated from the device lease; report that limitation. Do not treat a
CPU-only pass as device validation.

Report the checkout, target platform, environment path, PyPTO/simpler/ISA
commits, PTOAS and Torch versions, activation command, and exact verification
results. Explain any deviation from CI's toolchain and any unrun device step.
