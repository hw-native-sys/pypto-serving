# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the shared CI setup steps in temporary job environments."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from zipfile import ZipFile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTION = yaml.safe_load((ROOT / ".github/actions/setup-ci-job/action.yml").read_text())
BUILD = next(step for step in ACTION["runs"]["steps"]
             if step.get("name") == "Build and install pypto and simpler")
CHECK = next(step for step in ACTION["runs"]["steps"]
             if step.get("name") == "Check runner config and derive cache paths")
EXPECTED = {
    ("ci.yml", "unit-tests"): "a2a3",
    ("ci.yml", "v41-a5-smoke"): "a5",
    ("daily_ci.yml", "dspark-performance"): "a2a3",
    ("daily_v41_a5.yml", "v41-performance"): "a5",
}
CALLERS = {
    (path.name, job_name): step
    for path in (ROOT / ".github/workflows").glob("*.yml")
    for job_name, job in yaml.safe_load(path.read_text())["jobs"].items()
    for step in job.get("steps", [])
    if step.get("uses", "").endswith("/.github/actions/setup-ci-job")
}


def run_build(tmp_path, caller, *, retry=False):
    """Resolve actual input expressions, then execute the unmodified YAML run body."""
    inputs = {key: value.get("default", "") for key, value in ACTION["inputs"].items()}
    inputs.update(caller["with"])

    def resolve(value):
        return re.sub(r"\$\{\{ inputs\.([\w-]+) \}\}", lambda m: str(inputs[m[1]]), value)

    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    calls = tmp_path / "calls.jsonl"
    # All process/tool leaves record argv only. No real pip, compiler or cleanup
    # runs; the composite's shell control flow and argument forwarding are real.
    leaf = f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ["CALLS"], "a") as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
if name == "pip" and os.environ["PYPTO_SRC"] + "/runtime" in sys.argv[1:]:
    marker = Path(os.environ["CALLS"] + ".retried")
    if os.environ["FAIL_RUNTIME_ONCE"] == "1" and not marker.exists():
        marker.touch()
        sys.exit(1)
'''
    for name in ("pip", "python", "rm", "sleep"):
        tool = tool_dir / name
        tool.write_text(leaf)
        tool.chmod(0o755)
    (tmp_path / "activate.sh").write_text("")
    env = {
        "PATH": str(tool_dir) + os.pathsep + os.defpath,
        "CALLS": str(calls),
        "PYPTO_SRC": str(tmp_path / "pypto"),
        "FAIL_RUNTIME_ONCE": "1" if retry else "0",
        **{key: resolve(value) for key, value in BUILD.get("env", {}).items()},
    }
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", BUILD["run"]],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15,
    )
    records = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return result, records, env["PYPTO_SRC"]


def test_all_shared_setup_callers_have_platform_coverage():
    assert CALLERS.keys() == EXPECTED.keys()


def test_cached_ptoas_wheel_wins_over_same_version_on_pythonpath(tmp_path):
    python = sys.executable if sys.version_info[:2] == (3, 10) else shutil.which("python3.10")
    if not python:
        pytest.skip("the CI wheel requires CPython 3.10")

    version = "0.72"
    info = f"ptoas-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: ptoas\nVersion: {version}\n"
    cann = tmp_path / "cann"
    (cann / info).mkdir(parents=True)
    (cann / info / "METADATA").write_text(metadata)
    (cann / "ptoas").mkdir()
    (cann / "ptoas" / "__init__.py").write_text("")

    arch = platform.machine()
    tag = f"cp310-cp310-manylinux_2_34_{arch}"
    wheel = tmp_path / f"ptoas-{version}-{tag}.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("ptoas/__init__.py", "def main():\n    print(__file__)\n")
        archive.writestr(f"{info}/METADATA", metadata)
        archive.writestr(f"{info}/WHEEL", f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n")
        archive.writestr(f"{info}/entry_points.txt", "[console_scripts]\nptoas = ptoas:main\n")
        records = [*archive.namelist(), f"{info}/RECORD"]
        archive.writestr(f"{info}/RECORD", "".join(f"{name},,\n" for name in records))

    job = tmp_path / "job"
    subprocess.run([python, "-I", "-m", "venv", "--without-pip", str(job)], check=True)
    (tmp_path / "activate.sh").write_text(
        'source job/bin/activate\nexport PYTHONPATH="$PWD/cann${PYTHONPATH:+:$PYTHONPATH}"\n'
    )
    root = tmp_path / "ptoas-bin"
    install = next(step for step in ACTION["runs"]["steps"]
                   if step.get("name") == "Install ptoas (local cache)")
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", install["run"]], cwd=tmp_path,
        env={
            "PATH": os.defpath, "HOME": str(tmp_path), "PIP_NO_INDEX": "1",
            "PTOAS_ROOT": str(root), "PTOAS_CACHE_DIR": str(tmp_path),
            "PTOAS_ARCH": arch, "PTOAS_VERSION": f"v{version}",
            "PTOAS_SHA256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The wheel's CLI prints the imported module path after CI re-sources activate.sh.
    assert Path(result.stdout.splitlines()[-1]).is_relative_to(root)


@pytest.mark.parametrize("platform,ci_job", [("a2a3", "unit-tests"), ("a5", "v41-a5-smoke")])
def test_external_platform_selection_preserves_native_job_names(platform, ci_job):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    assert set(jobs) == {"pre-commit", "unit-tests", "v41-a5-smoke"}
    job = jobs[ci_job]
    assert "name" not in job and "uses" not in job
    assert jobs["pre-commit"]["if"] == "inputs.platform == ''"
    assert job["if"] == f"inputs.platform == '' || inputs.platform == '{platform}'"
    call = workflow[True]["workflow_call"]
    assert call["outputs"][f"{platform}-result"]["value"] == "${{ jobs." + ci_job + ".result }}"
    setup = CALLERS[("ci.yml", ci_job)]["with"]
    assert re.fullmatch(r"\$\{\{ inputs.pypto-revision \|\| '[a-f0-9]{40}' \}\}", setup["pypto-revision"])
    # A cross-repository call has the PyPTO caller in github.repository.
    # The action bootstrap and both serving checkout attempts need the callee.
    checkout_steps = [
        step for step in ACTION["runs"]["steps"] if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkout_steps) == 2
    for checkout in [job["steps"][0], *checkout_steps]:
        assert checkout["with"]["repository"] == "${{ job.workflow_repository }}"
        assert checkout["with"]["ref"] == "${{ job.workflow_sha }}"


@pytest.mark.parametrize("caller_key,platform", EXPECTED.items())
@pytest.mark.parametrize("retry", [False, True])
def test_job_platform_reaches_only_simpler_install(tmp_path, caller_key, platform, retry):
    result, calls, source = run_build(tmp_path, CALLERS[caller_key], retry=retry)
    assert result.returncode == 0, result.stdout + result.stderr
    compiler = [call for call in calls if call[:2] == ["pip", "install"] and call[-1] == source]
    runtime = [call for call in calls if call[:2] == ["pip", "install"] and source + "/runtime" in call[2:]]
    assert compiler == [["pip", "install", "--no-build-isolation", source]]
    assert runtime == [[
        "pip", "install", "--no-build-isolation",
        source + "/runtime", f"--config-settings=build.targets=build_package_{platform}",
    ]] * (2 if retry else 1)
    cleanup = [call for call in calls if call[0] == "rm"]
    assert cleanup == ([["rm", "-rf", source + "/runtime/build", source + "/runtime/_skbuild"]]
                       if retry else [])


@pytest.mark.parametrize("platform", ["", "a2a3sim"])
def test_invalid_runtime_platform_fails_before_install(tmp_path, platform):
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", CHECK["run"]],
        cwd=tmp_path,
        env={"PATH": os.defpath, "NEEDS_DEVICE": "true", "RUNTIME_PLATFORM": platform},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "runtime-platform" in result.stdout + result.stderr
