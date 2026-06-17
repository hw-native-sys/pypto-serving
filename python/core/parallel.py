# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass


_SUPPORTED_ROUTING_POLICIES = {"least_pending_tokens"}


@dataclass(frozen=True)
class ParallelConfig:
    """Serving parallelism contract for logical model replicas."""

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    expert_placement_strategy: str = "linear"
    all2all_backend: str = "none"
    devices: tuple[int, ...] = (0,)
    data_parallel_routing: str = "least_pending_tokens"

    def __post_init__(self) -> None:
        devices = tuple(int(device) for device in self.devices)
        object.__setattr__(self, "devices", devices)

        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        if self.pipeline_parallel_size != 1:
            raise ValueError("pipeline_parallel_size > 1 is not supported yet")
        if self.enable_expert_parallel:
            raise ValueError("expert parallel is not supported yet")
        if self.data_parallel_routing not in _SUPPORTED_ROUTING_POLICIES:
            supported = ", ".join(sorted(_SUPPORTED_ROUTING_POLICIES))
            raise ValueError(
                f"unsupported data_parallel_routing={self.data_parallel_routing!r}; "
                f"supported policies: {supported}"
            )
        if not devices:
            raise ValueError("devices must contain at least one device id")
        if len(set(devices)) != len(devices):
            raise ValueError(f"devices must not contain duplicates: {devices}")

        expected_devices = self.data_parallel_size * self.tensor_parallel_size
        if len(devices) != expected_devices:
            raise ValueError(
                "number of devices must equal data_parallel_size * tensor_parallel_size: "
                f"devices={len(devices)}, data_parallel_size={self.data_parallel_size}, "
                f"tensor_parallel_size={self.tensor_parallel_size}"
            )

    @property
    def replica_device_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return devices grouped by DP replica, each group being one TP group."""
        groups = []
        for dp_rank in range(self.data_parallel_size):
            start = dp_rank * self.tensor_parallel_size
            end = start + self.tensor_parallel_size
            groups.append(self.devices[start:end])
        return tuple(groups)

    def for_replica(self, device_group: tuple[int, ...]) -> "ParallelConfig":
        """Return a single-DP-replica view for one worker process."""
        return ParallelConfig(
            data_parallel_size=1,
            tensor_parallel_size=len(device_group),
            pipeline_parallel_size=self.pipeline_parallel_size,
            enable_expert_parallel=self.enable_expert_parallel,
            expert_placement_strategy=self.expert_placement_strategy,
            all2all_backend=self.all2all_backend,
            devices=device_group,
            data_parallel_routing=self.data_parallel_routing,
        )


def parse_device_ids(value: str | None, *, default_device: int = 0) -> tuple[int, ...]:
    """Parse a comma-separated device list, falling back to one default device."""
    if value is None or not value.strip():
        return (int(default_device),)
    parts = [part.strip() for part in value.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"invalid devices list: {value!r}")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"devices must be a comma-separated list of integers: {value!r}") from exc
