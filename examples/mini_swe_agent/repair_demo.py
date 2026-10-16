# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare and independently check the small log-report repair exercise."""

import argparse
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


FIXTURE = Path(__file__).resolve().parent / "tasks" / "log_report"
EXTRA_CHECKS = r"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from report import summarize

def row(n, status="ok", model="alpha"):
    return {"latency_ms": n, "status": status, "model": model}

def expected(total, ok, mean, p95):
    return {"requests": total, "successful": ok, "failed": total-ok, "mean_ms": mean, "p95_ms": p95}

cases = [
    ("empty", [], expected(0, 0, 0, 0)),
    ("fractional", [row(0.5), row(1.25)], expected(2, 2, 0.875, 1.25)),
    ("single", [row(7.5)], expected(1, 1, 7.5, 7.5)),
    ("rank_20", [row(n) for n in range(20, 0, -1)], expected(20, 20, 10.5, 19)),
    ("rank_21", [row(n) for n in range(21, 0, -1)], expected(21, 21, 11, 20)),
    ("failed_fast_request", [row(0, "error"), row(10), row(30)], expected(3, 2, 20, 30)),
]
for name, records, target in cases:
    actual = summarize(iter(records))
    if actual != target:
        raise AssertionError((name, actual, target))
    print("PASS", name)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "access.jsonl"
    path.write_text("\n" + "\n\n".join(json.dumps(r) for r in [row(10), row(30, model="alphabet"), row(5, "error")]) + "\n")
    for name, flags, target in [
        ("cli_all", [], expected(3, 2, 20, 30)),
        ("cli_exact", ["--model", "alpha"], expected(2, 1, 10, 10)),
        ("cli_no_match", ["--model", "missing"], expected(0, 0, 0, 0)),
    ]:
        result = subprocess.run([sys.executable, "cli.py", str(path), *flags], text=True, capture_output=True, check=True)
        actual = json.loads(result.stdout)
        if actual != target:
            raise AssertionError((name, actual, target))
        print("PASS", name)
"""


def run_tests(workspace, extra=False):
    command = [sys.executable, "-c", EXTRA_CHECKS] if extra else [sys.executable, "-m", "unittest", "-v"]
    try:
        result = subprocess.run(command, cwd=workspace, text=True, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        output = "".join(part.decode("utf-8", errors="replace") if isinstance(part, bytes) else (part or "")
                         for part in (error.stdout, error.stderr))
        return {"returncode": 124, "output": output + f"\nCheck timed out after {error.timeout} seconds.\n"}
    return {"returncode": result.returncode, "output": result.stdout + result.stderr}


def source_files(directory):
    return {path.relative_to(directory) for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.relative_to(directory).parts
            and path.suffix != ".pyc"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    check = commands.add_parser("check")
    check.add_argument("workspace", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        root = Path("outputs/mini_swe_agent_repair").resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="log_report_", dir=root))
        workspace = run_dir / "workspace"
        shutil.copytree(FIXTURE, workspace, ignore=shutil.ignore_patterns("__pycache__"))
        baseline = run_tests(workspace)
        (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
        print(f"Workspace: {workspace}")
        print(baseline["output"])
        return 0 if baseline["returncode"] == 1 else 1
    workspace = args.workspace.resolve()
    public = run_tests(workspace)
    extra = run_tests(workspace, extra=True)
    fixture_files = source_files(FIXTURE)
    unchanged = (
        source_files(workspace) == fixture_files
        and all((workspace / name).read_bytes() == (FIXTURE / name).read_bytes()
                for name in fixture_files - {Path("report.py"), Path("cli.py")})
    )
    patch = "".join("".join(difflib.unified_diff(
        (FIXTURE / name).read_text().splitlines(keepends=True),
        ((workspace / name).read_text() if (workspace / name).is_file() else "").splitlines(keepends=True),
        fromfile=f"before/{name}", tofile=f"after/{name}",
    )) for name in ("report.py", "cli.py"))
    summary = {"passed": unchanged and public["returncode"] == 0 and extra["returncode"] == 0,
               "tests_and_requirements_unchanged": unchanged, "public": public, "independent": extra}
    (workspace.parent / "verification.json").write_text(json.dumps(summary, indent=2))
    (workspace.parent / "changes.patch").write_text(patch)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
