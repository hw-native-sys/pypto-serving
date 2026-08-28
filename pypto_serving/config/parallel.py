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
_SUPPORTED_PLACEMENT_MODES = {"replica", "overlapped"}


@dataclass(frozen=True)
class ParallelConfig:
    """Serving parallelism contract for replica and model-local rank groups.

    ``replica`` placement keeps the historical serving contract: DP creates
    independent model replicas and each replica owns one TP device group.
    ``overlapped`` placement describes hybrid kernels whose DP, TP, and EP
    axes reuse the same physical ranks, as DeepSeekV4 does.
    """

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    expert_placement_strategy: str = "linear"
    all2all_backend: str = "none"
    devices: tuple[int, ...] = (0,)
    data_parallel_routing: str = "least_pending_tokens"
    expert_parallel_size: int = 1
    placement_mode: str = "replica"

    def __post_init__(self) -> None:
        devices = tuple(int(device) for device in self.devices)
        object.__setattr__(self, "devices", devices)

        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if self.expert_parallel_size < 1:
            raise ValueError("expert_parallel_size must be >= 1")
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        if self.pipeline_parallel_size != 1:
            raise ValueError("pipeline_parallel_size > 1 is not supported yet")
        if self.placement_mode not in _SUPPORTED_PLACEMENT_MODES:
            supported = ", ".join(sorted(_SUPPORTED_PLACEMENT_MODES))
            raise ValueError(
                f"unsupported placement_mode={self.placement_mode!r}; supported modes: {supported}"
            )
        if self.placement_mode == "replica" and (
            self.enable_expert_parallel or self.expert_parallel_size != 1
        ):
            raise ValueError("expert parallel requires overlapped placement")
        if self.placement_mode == "overlapped":
            if self.enable_expert_parallel != (self.expert_parallel_size > 1):
                raise ValueError(
                    "enable_expert_parallel must match expert_parallel_size > 1 "
                    "for overlapped placement"
                )
            dense_world = self.data_parallel_size * self.tensor_parallel_size
            expert_world = self.expert_parallel_size
            if dense_world != self.worker_group_size or expert_world not in (
                1,
                self.worker_group_size,
            ):
                raise ValueError(
                    "overlapped placement requires DP * TP to span the worker group and "
                    "EP to be either 1 or the full worker group: "
                    f"DP={self.data_parallel_size}, TP={self.tensor_parallel_size}, "
                    f"EP={self.expert_parallel_size}, world={self.worker_group_size}"
                )
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

        expected_devices = self.num_replicas * self.worker_group_size
        if len(devices) != expected_devices:
            raise ValueError(
                "number of devices does not match the parallel placement: "
                f"devices={len(devices)}, replicas={self.num_replicas}, "
                f"worker_group_size={self.worker_group_size}, placement_mode={self.placement_mode!r}"
            )

    @property
    def num_replicas(self) -> int:
        """Return the number of independently routed serving replicas."""
        if self.placement_mode == "overlapped":
            return 1
        return self.data_parallel_size

    @property
    def worker_group_size(self) -> int:
        """Return the number of devices owned by one serving worker."""
        if self.placement_mode == "overlapped":
            return max(
                self.data_parallel_size * self.tensor_parallel_size,
                self.expert_parallel_size,
            )
        return self.tensor_parallel_size

    @property
    def replica_device_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return devices grouped by independently routed serving replica."""
        groups = []
        for replica_rank in range(self.num_replicas):
            start = replica_rank * self.worker_group_size
            end = start + self.worker_group_size
            groups.append(self.devices[start:end])
        return tuple(groups)

    def for_replica(self, device_group: tuple[int, ...]) -> "ParallelConfig":
        """Return the parallel topology visible to one serving worker."""
        if self.placement_mode == "overlapped":
            return ParallelConfig(
                data_parallel_size=self.data_parallel_size,
                tensor_parallel_size=self.tensor_parallel_size,
                expert_parallel_size=self.expert_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                enable_expert_parallel=self.enable_expert_parallel,
                expert_placement_strategy=self.expert_placement_strategy,
                all2all_backend=self.all2all_backend,
                devices=device_group,
                data_parallel_routing=self.data_parallel_routing,
                placement_mode=self.placement_mode,
            )
        return ParallelConfig(
            data_parallel_size=1,
            tensor_parallel_size=len(device_group),
            expert_parallel_size=1,
            pipeline_parallel_size=self.pipeline_parallel_size,
            enable_expert_parallel=self.enable_expert_parallel,
            expert_placement_strategy=self.expert_placement_strategy,
            all2all_backend=self.all2all_backend,
            devices=device_group,
            data_parallel_routing=self.data_parallel_routing,
            placement_mode=self.placement_mode,
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
