# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate complete text actions before passing them to the shell."""

import shlex

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


class CompleteTextModel(LitellmTextbasedModel):
    """Reject incomplete generations and unsupported printed action wrappers."""

    def _parse_actions(self, response) -> list[dict]:
        choice = response.choices[0]
        # LiteLLM normalizes unknown reasons (including aborted) to stop.
        native_fields = getattr(choice, "provider_specific_fields", None) or {}
        reason = native_fields.get("native_finish_reason", choice.finish_reason)
        if reason not in ("eos", "stop"):
            raise FormatError({
                "role": "user",
                "content": f"Generation ended with {reason!r}; no command was executed. Retry concisely.",
                "extra": {"interrupt_type": "FormatError", "finish_reason": reason},
            })
        actions = super()._parse_actions(response)
        command = actions[0]["command"]
        try:
            words = shlex.split(command)
        except ValueError:
            # This check recognizes a specific wrapper, not general shell syntax.
            return actions
        is_wrapper = (
            len(words) >= 3 and words[0] == "printf" and words[2] == "RUN_SHELL_COMMAND"
        ) or (
            len(words) >= 2 and words[0] == "echo" and words[1] == "RUN_SHELL_COMMAND"
        )
        if is_wrapper:
            raise FormatError({
                "role": "user",
                "content": (
                    "Nothing was executed. RUN_SHELL_COMMAND is not a supported protocol. "
                    "Your command would only print text, not run the printed command. "
                    "Put the actual shell command directly in one bash code block, for example "
                    "cat /path/to/README.md. Wait for real file contents before summarizing."
                ),
                "extra": {"interrupt_type": "FormatError", "rejected_command": command},
            })
        return actions
