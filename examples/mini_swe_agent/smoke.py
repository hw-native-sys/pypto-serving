# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify a real text-action round trip using mini-swe-agent and PyPTO Serving."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import shlex
import tempfile
import time

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment

from client.model import CompleteTextModel


FINISH_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SYSTEM_TEMPLATE = """You operate a shell through text actions.
Every reply must contain exactly one command block in this format:
```bash
your command
```
Use no other code blocks. Be concise. Wait for the command observation before deciding the next action.
After reading the requested file, finish by running exactly:
printf '%s\\n' 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' 'THE_EXACT_FILE_CONTENT'
Replace THE_EXACT_FILE_CONTENT with the observed single line. Do not guess the file content.
"""


class SmokeEnvironment(LocalEnvironment):
    """Only execute the two commands needed for this bounded integration probe."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commands: list[str] = []

    def execute(self, action: dict, **kwargs) -> dict:
        command = action.get("command", "").strip()
        if command != "cat input.txt":
            parts = shlex.split(command)
            valid_submission = (
                len(parts) == 4
                and parts[:3] == ["printf", "%s\\n", FINISH_MARKER]
                and command == f"printf '%s\\n' '{FINISH_MARKER}' {shlex.quote(parts[3])}"
            )
            # Also accept the quoted form requested in the prompt, only for an alphanumeric result.
            if len(parts) == 4 and parts[:3] == ["printf", "%s\\n", FINISH_MARKER]:
                valid_submission |= (
                    parts[3].replace("_", "").isalnum()
                    and command == f"printf '%s\\n' '{FINISH_MARKER}' '{parts[3]}'"
                )
            if not valid_submission:
                raise ValueError(f"Unexpected smoke-test command: {command!r}")
        self.commands.append(command)
        print(f"ACTION {len(self.commands)}: {command}", flush=True)
        return super().execute(action, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8899/v1")
    parser.add_argument("--model", default="qwen3-agent")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mini_swe_agent_step1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="roundtrip_", dir=args.output_dir.resolve()))
    workspace = run_dir / "workspace"
    workspace.mkdir()
    expected = f"hello_{secrets.token_hex(6)}"
    (workspace / "input.txt").write_text(expected + "\n")

    os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"]))
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    model = CompleteTextModel(
        model_name=f"openai/{args.model}",
        action_regex=r"```bash\s*\n(.*?)\n```",
        cost_tracking="ignore_errors",
        model_kwargs={
            "api_base": args.api_base,
            "api_key": "local-placeholder",
            "temperature": 0.0,
            "max_tokens": 256,
            "stream": False,
            "timeout": 120,
            "num_retries": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )
    environment = SmokeEnvironment(cwd=str(workspace), timeout=10)
    agent = DefaultAgent(
        model,
        environment,
        system_template=SYSTEM_TEMPLATE,
        instance_template="{{task}}",
        step_limit=4,
        cost_limit=0,
        wall_time_limit_seconds=300,
        max_consecutive_format_errors=2,
        output_path=run_dir / "trajectory.json",
    )
    started = time.monotonic()
    result = agent.run("First run exactly cat input.txt. Then submit its exact single-line content.")
    responses = [message["extra"]["response"] for message in agent.messages
                 if "response" in message.get("extra", {})]
    observations = [message.get("extra", {}).get("raw_output") for message in agent.messages]
    passed = (
        result.get("exit_status") == "Submitted"
        and result.get("submission", "").strip() == expected
        and expected + "\n" in observations
        and environment.commands[0:1] == ["cat input.txt"]
        and agent.n_calls >= 2
    )
    summary = {
        "passed": passed,
        "scope": "Bounded text-action smoke test; not a general coding-agent evaluation",
        "versions": {name: importlib.metadata.version(name) for name in ("mini-swe-agent", "litellm")},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "model_calls": agent.n_calls,
        "commands": environment.commands,
        "expected": expected,
        "result": result,
        "usage": {key: sum((response.get("usage") or {}).get(key, 0) or 0 for response in responses)
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "trajectory": str(run_dir / "trajectory.json"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
