# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sampling capabilities shared by request admission and worker routing."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SamplingCapabilities:
    prefill_stop_mask: Literal["host", "device"] = "host"
    decode_stop_mask: Literal["host", "device"] = "host"
    honor_min_tokens: bool = True

    def stop_mask_on_device(self, phase: Literal["prefill", "decode"]) -> bool:
        return getattr(self, f"{phase}_stop_mask") == "device"


def sampling_capabilities(executor_cls: str, num_speculative_tokens: int) -> SamplingCapabilities:
    """Declare the stop-mask path for each serving executor configuration."""
    if executor_cls == "PyptoDeepSeekV4DSparkExecutor":
        return SamplingCapabilities(prefill_stop_mask="device", decode_stop_mask="device")
    if executor_cls == "PyptoDeepSeekV4Executor":
        # The V4 MTP fused sampler does not yet accept a stop-token mask.
        return SamplingCapabilities(honor_min_tokens=num_speculative_tokens == 0)
    if executor_cls in ("PyptoQwen14BExecutor", "PyptoDeepSeekV41Executor"):
        return SamplingCapabilities()
    raise ValueError(f"sampling capabilities are not declared for {executor_cls!r}")
