# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise completion checks using real files, shells, and verification commands."""

import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

from minisweagent.exceptions import Submitted

from runtime import VerifiedEnvironment, coding_config, shell_environment


COMPLETE = {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}


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

    def test_editing_tests_does_not_bypass_acceptance(self):
        self.env.execute({"command": "printf fixed > source.txt; printf altered > test.txt"})
        output = self.env.execute(COMPLETE)
        self.assertEqual(output["returncode"], 1)
        self.assertIn("Protected file was modified: test.txt", output["output"])
        self.assertIn("checks passed", output["output"])

    def test_silent_sed_noop_reports_no_change(self):
        output = self.env.execute({"command": "sed -i 's/not_present/fixed/' source.txt"})
        self.assertEqual(output["returncode"], 0)
        self.assertEqual(output["extra"]["changed_files"], [])
        self.assertIn("The edit made no changes", output["output"])
        self.assertEqual((self.root / "source.txt").read_text(), "broken")

    def test_checker_cannot_fix_the_candidate_and_accept_it(self):
        self.env.config.verify_command = "printf fixed > source.txt"
        output = self.env.execute(COMPLETE)
        self.assertEqual(output["returncode"], 1)
        self.assertIn("Verification modified tracked files", output["output"])

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

    def test_official_coding_workflow_and_consistent_text_actions(self):
        config = coding_config()
        self.assertIn("Edit the source code", config["agent"]["instance_template"])
        self.assertIn("Verify your fix", config["agent"]["instance_template"])
        self.assertIn("cat <<'EOF'", config["agent"]["instance_template"])
        self.assertNotIn("mswea_bash_command", json.dumps(config))


if __name__ == "__main__":
    unittest.main()
