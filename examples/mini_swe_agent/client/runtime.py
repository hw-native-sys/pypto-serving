# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Use upstream coding prompts and verify completion against local evidence."""

import hashlib
import json
import os
from pathlib import Path
import sys

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.exceptions import Submitted


def coding_config() -> dict:
    config = get_config_from_spec(builtin_config_dir / "default.yaml")
    for section, key in (("agent", "system_template"), ("agent", "instance_template"),
                         ("model", "format_error_template")):
        config[section][key] = config[section][key].replace("mswea_bash_command", "bash")
    config["agent"]["system_template"] += (
        "\nReply in the user's language. Explain the next action briefly. "
        "Commands in bash blocks execute literally; do not print RUN_SHELL_COMMAND wrappers. "
        "A successful shell exit does not prove a replacement changed a file. "
        "After editing, inspect the result and run the tests before claiming success. "
        "If an edit fails, inspect the affected lines instead of repeating the same command. "
        "Read the project's requirements before editing; do not guess behavior from one assertion. "
        "For Python source edits, prefer literal replacements with an assertion over sed regular "
        "expressions (brackets and asterisks in code change regex meaning). For example:\n"
        "```bash\npython - <<'PY'\nfrom pathlib import Path\np = Path('module.py')\n"
        "text = p.read_text()\nold = 'return old_value'\nnew = 'return new_value'\n"
        "assert text.count(old) == 1, 'Expected exactly one target; inspect the file again'\n"
        "p.write_text(text.replace(old, new))\nPY\n```\n"
    )
    return config


def shell_environment(extra: dict | None = None) -> dict:
    return dict(extra or {}) | {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "VIRTUAL_ENV": sys.prefix,
        "PYTHONNOUSERSITE": "1",
    }


class VerifiedEnvironmentConfig(LocalEnvironmentConfig):
    verify_command: str = ""
    require_changes: list[str] = []
    protected_files: list[str] = []
    verification_dir: str = ""
    verification_timeout: int = 60


class VerifiedEnvironment(LocalEnvironment):
    """Reject completion until configured file checks and tests actually pass."""

    def __init__(self, **kwargs):
        super().__init__(config_class=VerifiedEnvironmentConfig, **kwargs)
        if (self.config.require_changes or self.config.protected_files) and not self.config.verify_command:
            raise ValueError("File acceptance checks require a verification command")
        self.workspace = Path(self.config.cwd).resolve()
        self.paths = {}
        for name in self.config.require_changes + self.config.protected_files:
            path = (self.workspace / name).resolve()
            if not path.is_relative_to(self.workspace):
                raise ValueError(f"Acceptance path must stay inside the workspace: {name}")
            self.paths[name] = path
        self.initial = self._snapshot()
        self.verifications: list[dict] = []

    def _snapshot(self) -> dict:
        return {name: hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                for name, path in self.paths.items()}

    def execute(self, action: dict, **kwargs) -> dict:
        before = self._snapshot()
        try:
            output = super().execute(action, **kwargs)
        except Submitted as completed:
            if not self.config.verify_command:
                raise
            verification = self.verify()
            if verification["passed"]:
                completed.messages[0]["extra"]["verification"] = verification
                print("Completion verified: required file changes and checks passed.", flush=True)
                raise
            print("Completion rejected: verification failed; returning evidence to the Agent.", flush=True)
            return {
                "returncode": 1,
                "output": "Completion rejected. Fix the remaining issues and rerun tests.\n"
                          + "\n".join(verification["reasons"]) + "\n" + verification["output"],
                "exception_info": "",
                "extra": {"verification": verification},
            }
        after = self._snapshot()
        if self.paths:
            changed = [name for name in self.paths if before[name] != after[name]]
            output["output"] += "\n[Tracked files changed by this command: " + (", ".join(changed) or "none") + "]\n"
            if not changed and "sed -i" in action.get("command", ""):
                output["output"] += (
                    "The edit made no changes. sed can exit zero without matching anything. "
                    "Inspect the exact source and use a literal Python replacement with an assertion; "
                    "do not claim this edit succeeded or repeat it unchanged.\n"
                )
            output.setdefault("extra", {})["changed_files"] = changed
        return output

    def verify(self) -> dict:
        before = self._snapshot()
        checker = LocalEnvironment(cwd=str(self.workspace), env=self.config.env,
                                   timeout=self.config.verification_timeout)
        try:
            output = checker.execute({"command": self.config.verify_command})
        except Submitted:
            output = {"returncode": 1, "output": "Verification command emitted an Agent completion marker."}
        after = self._snapshot()
        reasons = []
        for name in self.config.require_changes:
            if after[name] == self.initial[name]:
                reasons.append(f"Required source file is unchanged: {name}")
        for name in self.config.protected_files:
            if after[name] != self.initial[name]:
                reasons.append(f"Protected file was modified: {name}")
        if before != after:
            reasons.append("Verification modified tracked files; recheck the candidate before submitting.")
        if output["returncode"] != 0:
            reasons.append(f"Verification command failed with exit code {output['returncode']}.")
        verification = {
            "passed": not reasons, "command": self.config.verify_command,
            "returncode": output["returncode"], "output": output["output"],
            "reasons": reasons, "initial_files": self.initial, "current_files": after,
        }
        self.verifications.append(verification)
        if self.config.verification_dir:
            folder = Path(self.config.verification_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"verification_{len(self.verifications)}.json").write_text(json.dumps(verification, indent=2))
        return verification
