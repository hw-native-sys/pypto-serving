# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify history selection without requiring a model server or model files."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers

from budget import BudgetedTextModel, ContextBudgetExceeded


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


if __name__ == "__main__":
    unittest.main()
