# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check copyright headers in git-tracked source files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY_HEADER = """\
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------"""


def get_git_tracked_files(root: Path) -> list[Path]:
    """Return git-tracked files under ``root``."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        root / line
        for line in result.stdout.strip().split("\n")
        if line and (root / line).is_file()
    ]


def check_header(path: Path) -> tuple[bool, str]:
    """Check that ``path`` starts with the project Python header."""
    expected_lines = PY_HEADER.split("\n")
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return False, f"Error reading: {exc}"
    if not content:
        return False, "File is empty"
    actual_lines = content.split("\n")
    if len(actual_lines) < len(expected_lines):
        return False, f"Too few lines ({len(actual_lines)} < {len(expected_lines)})"
    for index, expected in enumerate(expected_lines):
        if actual_lines[index] != expected:
            return False, (
                f"Line {index + 1}: expected: {expected!r}, "
                f"got: {actual_lines[index]!r}"
            )
    return True, ""


def main() -> int:
    """Run the header check."""
    root = Path(__file__).resolve().parent.parent.parent
    all_files = get_git_tracked_files(root)
    py_files = [path for path in all_files if path.suffix == ".py"]

    if not py_files:
        print("No Python files to check.")
        return 0

    print(f"Checking {len(py_files)} file(s)...")
    failed = []
    for path in py_files:
        ok, message = check_header(path)
        if not ok:
            failed.append((path, message))
            print(f"  FAIL {path.relative_to(root)}: {message}")

    if failed:
        print(f"\n{len(failed)} file(s) missing or incorrect copyright header.")
        return 1

    print("All files have correct headers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
