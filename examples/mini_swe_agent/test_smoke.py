# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks for the integration's action execution boundaries."""

import tempfile
import unittest
from pathlib import Path

from litellm import ModelResponse
from minisweagent.exceptions import FormatError

from smoke import CompleteTextModel, SmokeEnvironment


class ActionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.model = CompleteTextModel(
            model_name="openai/qwen3-agent",
            cost_tracking="ignore_errors",
            action_regex=r"```bash\s*\n(.*?)\n```",
        )

    def response(self, content, reason):
        return ModelResponse(choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": reason,
        }])

    def test_success_reasons_accept_one_command(self):
        for reason in ("eos", "stop"):
            with self.subTest(reason=reason):
                parsed = self.model._parse_actions(self.response("```bash\ncat input.txt\n```", reason))
                self.assertEqual(parsed, [{"command": "cat input.txt"}])

    def test_incomplete_generation_rejects_even_complete_code_block(self):
        for reason in ("length", "aborted"):
            with self.subTest(reason=reason), self.assertRaises(FormatError):
                self.model._parse_actions(self.response("```bash\ncat input.txt\n```", reason))

    def test_missing_and_multiple_actions_rejected(self):
        for content in ("No action", "```bash\ncat input.txt\n```\n```bash\ncat input.txt\n```"):
            with self.subTest(content=content), self.assertRaises(FormatError):
                self.model._parse_actions(self.response(content, "stop"))

    def test_printed_action_wrapper_is_rejected_before_execution(self):
        for command in (
            "printf '%s\\n' 'RUN_SHELL_COMMAND' 'cat README.md'",
            'echo RUN_SHELL_COMMAND "cat README.md"',
        ):
            with self.subTest(command=command), self.assertRaises(FormatError) as caught:
                self.model._parse_actions(self.response(f"```bash\n{command}\n```", "stop"))
            self.assertIn("Nothing was executed", caught.exception.messages[0]["content"])
            self.assertEqual(caught.exception.messages[0]["extra"]["rejected_command"], command)

    def test_real_commands_and_completion_still_parse(self):
        for command in (
            "cat README.md",
            "rg RUN_SHELL_COMMAND README.md",
            "printf '%s\\n' 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' 'Done'",
        ):
            with self.subTest(command=command):
                self.assertEqual(
                    self.model._parse_actions(self.response(f"```bash\n{command}\n```", "stop")),
                    [{"command": command}],
                )

    def test_shell_chain_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "input.txt").write_text("probe\n")
            environment = SmokeEnvironment(cwd=directory)
            with self.assertRaises(ValueError):
                environment.execute({"command": "cat input.txt; touch unexpected"})
            self.assertFalse((workspace / "unexpected").exists())
            self.assertEqual(environment.commands, [])
            self.assertEqual(environment.execute({"command": "cat input.txt"})["output"], "probe\n")


if __name__ == "__main__":
    unittest.main()
