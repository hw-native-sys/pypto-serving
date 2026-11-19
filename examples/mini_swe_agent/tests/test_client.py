# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check text actions, context budgets, and completion acceptance on CPU."""

import copy
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

from litellm import ModelResponse
from minisweagent.exceptions import FormatError, Submitted
from tokenizers import Tokenizer, models, pre_tokenizers

from client.budget import BudgetedTextModel, ContextBudgetExceeded
from client.model import CompleteTextModel
from client.runtime import VerifiedEnvironment, shell_environment


COMPLETE = {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}


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

    def test_complete_commands_parse_for_successful_finish_reasons(self):
        for reason in ("eos", "stop"):
            for command in (
                "cat README.md",
                "rg RUN_SHELL_COMMAND README.md",
                "printf '%s\\n' 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' 'Done'",
            ):
                with self.subTest(reason=reason, command=command):
                    parsed = self.model._parse_actions(self.response(f"```bash\n{command}\n```", reason))
                    self.assertEqual(parsed, [{"command": command}])

    def test_incomplete_or_malformed_responses_produce_no_action(self):
        block = "```bash\ncat input.txt\n```"
        cases = (("length", block), ("aborted", block), ("stop", "No action"), ("stop", block + "\n" + block))
        for reason, content in cases:
            with self.subTest(reason=reason, content=content), self.assertRaises(FormatError):
                self.model._parse_actions(self.response(content, reason))

    def test_printed_action_wrapper_is_rejected_before_execution(self):
        for command in (
            "printf '%s\\n' 'RUN_SHELL_COMMAND' 'cat README.md'",
            'echo RUN_SHELL_COMMAND "cat README.md"',
        ):
            with self.subTest(command=command), self.assertRaises(FormatError) as caught:
                self.model._parse_actions(self.response(f"```bash\n{command}\n```", "stop"))
            self.assertIn("Nothing was executed", caught.exception.messages[0]["content"])
            self.assertEqual(caught.exception.messages[0]["extra"]["rejected_command"], command)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(folder / "tokenizer.json"))
        (folder / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}assistant "
        }))
        self.model = BudgetedTextModel(tokenizer_path=folder, context_limit=100,
                                      model_name="openai/test", model_kwargs={"max_tokens": 20})

    def messages(self):
        result = [{"role": "system", "content": "Keep requirements"},
                  {"role": "user", "content": "Repair the code without editing tests"}]
        for index in range(8):
            result.extend([
                {"role": "assistant", "content": f"command {index}"},
                {"role": "user", "content": "observed " * 20, "extra": {"raw_output": "observed " * 20}},
            ])
        return result

    def test_discards_old_pairs_and_preserves_task_and_latest_evidence(self):
        messages = self.messages()
        original = copy.deepcopy(messages)
        prepared = self.model._prepare_messages_for_api(messages)
        self.assertEqual(messages, original)
        self.assertEqual(prepared[:2], original[:2])
        self.assertEqual(prepared[-2]["content"], "command 7")
        self.assertEqual(prepared[-1]["content"], original[-1]["content"])
        self.assertLessEqual(self.model.counter.count(prepared) + 20, 100)
        dropped = self.model.context_events[-1]["dropped_message_indices"]
        self.assertTrue(dropped)
        for index in range(2, len(messages), 2):
            self.assertEqual(index in dropped, index + 1 in dropped)

    def test_user_followup_is_not_dropped(self):
        messages = self.messages()
        followup = {"role": "user", "content": "Also preserve the public API",
                    "extra": {"interrupt_type": "UserNewTask"}}
        messages.insert(6, followup)
        prepared = self.model._prepare_messages_for_api(messages)
        self.assertIn(followup["content"], [m["content"] for m in prepared])

    def test_large_observation_is_marked_and_original_retained(self):
        messages = self.messages()[:4]
        messages[-1]["content"] = "large output " * 1000
        original = messages[-1]["content"]
        prepared = self.model._prepare_messages_for_api(messages)
        self.assertLessEqual(self.model.counter.count(prepared) + 20, 100)
        self.assertIn("Observation shortened", prepared[-1]["content"])
        self.assertEqual(messages[-1]["content"], original)

    def test_oversized_task_is_rejected_instead_of_silently_shortened(self):
        with self.assertRaises(ContextBudgetExceeded):
            self.model._prepare_messages_for_api([
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "important requirement " * 100},
            ])


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "source.txt").write_text("broken")
        (self.root / "test.txt").write_text("original tests")
        check = "from pathlib import Path; assert Path('source.txt').read_text() == 'fixed'; print('checks passed')"
        self.env = VerifiedEnvironment(
            cwd=str(self.root), env=shell_environment(),
            verify_command=shlex.quote(sys.executable) + " -c " + shlex.quote(check),
            require_changes=["source.txt"], protected_files=["test.txt"],
            verification_dir=str(self.root / "audit"),
        )

    def test_false_completion_returns_evidence_and_audit(self):
        output = self.env.execute(COMPLETE)
        self.assertEqual(output["returncode"], 1)
        self.assertIn("unchanged: source.txt", output["output"])
        self.assertIn("AssertionError", output["output"])
        record = json.loads((self.root / "audit/verification_1.json").read_text())
        self.assertFalse(record["passed"])

    def test_actual_fix_passes_but_later_regression_is_rechecked(self):
        output = self.env.execute({"command": "printf fixed > source.txt"})
        self.assertEqual(output["extra"]["changed_files"], ["source.txt"])
        with self.assertRaises(Submitted) as caught:
            self.env.execute(COMPLETE)
        self.assertTrue(caught.exception.messages[0]["extra"]["verification"]["passed"])
        self.env.execute({"command": "printf regression > source.txt"})
        output = self.env.execute(COMPLETE)
        self.assertEqual(output["returncode"], 1)
        self.assertIn("AssertionError", output["output"])
        self.assertEqual(len(self.env.verifications), 2)

    def test_protected_files_and_checker_side_effects_reject_completion(self):
        for checker_changes_source in (False, True):
            with self.subTest(checker_changes_source=checker_changes_source):
                self.env.execute({"command": "printf fixed > source.txt; printf 'original tests' > test.txt"})
                if checker_changes_source:
                    self.env.config.verify_command = "printf checked > source.txt"
                    reason = "Verification modified tracked files"
                else:
                    self.env.execute({"command": "printf altered > test.txt"})
                    reason = "Protected file was modified: test.txt"
                output = self.env.execute(COMPLETE)
                self.assertEqual(output["returncode"], 1)
                self.assertIn(reason, output["output"])

    def test_silent_sed_noop_reports_no_change(self):
        output = self.env.execute({"command": "sed -i 's/not_present/fixed/' source.txt"})
        self.assertEqual(output["returncode"], 0)
        self.assertEqual(output["extra"]["changed_files"], [])
        self.assertIn("The edit made no changes", output["output"])
        self.assertEqual((self.root / "source.txt").read_text(), "broken")

    def test_shell_python_is_the_agent_interpreter(self):
        env = VerifiedEnvironment(cwd=str(self.root), env=shell_environment())
        output = env.execute({"command": 'python -c "import sys; print(sys.executable)"'})
        self.assertEqual(output["returncode"], 0)
        self.assertEqual(output["output"].strip(), sys.executable)
        with self.assertRaises(Submitted):
            env.execute(COMPLETE)

    def test_verification_timeout_is_a_rejection(self):
        self.env.config.verification_timeout = 1
        self.env.config.verify_command = shlex.quote(sys.executable) + " -c 'import time; time.sleep(5)'"
        output = self.env.execute(COMPLETE)
        self.assertEqual(output["returncode"], 1)
        self.assertFalse(output["extra"]["verification"]["passed"])


if __name__ == "__main__":
    unittest.main()
