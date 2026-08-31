# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_MIB = 1024 * 1024
_SIZE_UNITS = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
}


def _parse_size(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a byte count or size string")
    if isinstance(value, int):
        size = value
    elif isinstance(value, str):
        normalized = value.strip().lower()
        unit = "b"
        for candidate in ("gb", "mb", "kb", "b"):
            if normalized.endswith(candidate):
                normalized = normalized[: -len(candidate)].strip()
                unit = candidate
                break
        try:
            size = int(float(normalized) * _SIZE_UNITS[unit])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} has invalid size {value!r}") from exc
    else:
        raise ValueError(f"{field_name} must be a byte count or size string")
    if size < 0:
        raise ValueError(f"{field_name} must not be negative")
    return size


def _require_string(data: Mapping[str, Any], name: str, *, default: str | None = None) -> str:
    value = data.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"external prefix cache config requires non-empty {name!r}")
    return value.strip()


def _positive_int(data: Mapping[str, Any], name: str, *, default: int) -> int:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"external cache {name} must be a positive integer")
    return value


def _ascend_buffer_pool(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Mooncake ascend_buffer_pool must use 'buffer_count:size_mb' format")
    value = value.strip()
    if not value:
        return ""
    parts = value.split(":")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise ValueError("Mooncake ascend_buffer_pool must use 'buffer_count:size_mb' format")
    buffer_count, size_mb = (int(part) for part in parts)
    if (buffer_count == 0) != (size_mb == 0):
        raise ValueError("Mooncake ascend_buffer_pool values must both be zero or both be positive")
    return f"{buffer_count}:{size_mb}"


@dataclass(frozen=True, slots=True)
class MooncakeClientConfig:
    """Serializable Mooncake Store requester configuration."""

    metadata_server: str
    master_server_address: str
    protocol: str = "ascend"
    device_name: str = ""
    local_hostname: str = ""
    global_segment_size: int = 0
    local_buffer_size: int = 64 * _MIB
    enable_ssd_offload: bool = False
    ssd_offload_path: str = ""
    tenant_id: str = "default"
    ascend_buffer_pool: str = ""

    def __post_init__(self) -> None:
        normalized_buffer_pool = _ascend_buffer_pool(self.ascend_buffer_pool)
        object.__setattr__(self, "ascend_buffer_pool", normalized_buffer_pool)
        if not self.metadata_server or not self.master_server_address:
            raise ValueError("Mooncake metadata and master server addresses must not be empty")
        if not self.protocol:
            raise ValueError("Mooncake protocol must not be empty")
        if self.global_segment_size < 0:
            raise ValueError("Mooncake global_segment_size must not be negative")
        if self.local_buffer_size <= 0:
            raise ValueError("Mooncake local_buffer_size must be positive")
        if not self.tenant_id:
            raise ValueError("Mooncake tenant_id must not be empty")
        if self.enable_ssd_offload:
            if not self.ssd_offload_path:
                raise ValueError(
                    "Mooncake enable_ssd_offload requires a non-empty ssd_offload_path"
                )
            if not os.path.isabs(self.ssd_offload_path):
                raise ValueError("Mooncake ssd_offload_path must be an absolute path")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MooncakeClientConfig":
        allowed = {
            "ascend_buffer_pool",
            "device_name",
            "enable_ssd_offload",
            "global_segment_size",
            "local_buffer_size",
            "local_hostname",
            "master_server_address",
            "metadata_server",
            "protocol",
            "ssd_offload_path",
            "tenant_id",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("unknown Mooncake config fields: " + ", ".join(unknown))
        enable_ssd_offload = data.get("enable_ssd_offload", False)
        if not isinstance(enable_ssd_offload, bool):
            raise ValueError("Mooncake enable_ssd_offload must be a boolean")
        return cls(
            metadata_server=_require_string(data, "metadata_server"),
            master_server_address=_require_string(data, "master_server_address"),
            protocol=_require_string(data, "protocol", default="ascend"),
            device_name=str(data.get("device_name", "")).strip(),
            local_hostname=str(data.get("local_hostname", "")).strip(),
            global_segment_size=_parse_size(
                data.get("global_segment_size", 0), field_name="global_segment_size"
            ),
            local_buffer_size=_parse_size(
                data.get("local_buffer_size", 64 * _MIB), field_name="local_buffer_size"
            ),
            enable_ssd_offload=enable_ssd_offload,
            ssd_offload_path=str(data.get("ssd_offload_path", "")).strip(),
            tenant_id=_require_string(data, "tenant_id", default="default"),
            ascend_buffer_pool=_ascend_buffer_pool(data.get("ascend_buffer_pool")),
        )


@dataclass(frozen=True, slots=True)
class ExternalPrefixCacheConfig:
    """DeepSeek external prefix-cache policy shared by engine and worker."""

    mooncake: MooncakeClientConfig
    model_revision: str
    tokenizer_revision: str
    min_tokens: int = 1024
    load_timeout_ms: int = 30_000
    transfer_concurrency: int = 2
    failure_policy: str = "cold_miss"
    enable_save: bool = True
    backend: str = "mooncake"

    def __post_init__(self) -> None:
        if self.backend != "mooncake":
            raise ValueError(f"unsupported external prefix cache backend {self.backend!r}")
        if self.mooncake.protocol != "ascend":
            raise ValueError("DeepSeek external prefix caching requires Mooncake protocol 'ascend'")
        if not self.model_revision or not self.tokenizer_revision:
            raise ValueError("external cache model/tokenizer revisions must not be empty")
        if self.min_tokens <= 0:
            raise ValueError("external cache min_tokens must be positive")
        if self.load_timeout_ms <= 0:
            raise ValueError("external cache load_timeout_ms must be positive")
        if self.transfer_concurrency <= 0:
            raise ValueError("external cache transfer_concurrency must be positive")
        if self.failure_policy not in ("cold_miss", "fail_startup"):
            raise ValueError("external cache failure_policy must be 'cold_miss' or 'fail_startup'")

    @classmethod
    def from_file(cls, path: str | Path) -> "ExternalPrefixCacheConfig":
        config_path = Path(path)
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read external prefix cache config {config_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("external prefix cache config must be a JSON object")
        allowed = {
            "backend",
            "enable_save",
            "failure_policy",
            "load_timeout_ms",
            "min_tokens",
            "model_revision",
            "mooncake",
            "tokenizer_revision",
            "transfer_concurrency",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("unknown external prefix cache config fields: " + ", ".join(unknown))
        mooncake_data = data.get("mooncake")
        if not isinstance(mooncake_data, dict):
            raise ValueError("external prefix cache config requires a 'mooncake' object")
        enable_save = data.get("enable_save", True)
        if not isinstance(enable_save, bool):
            raise ValueError("external cache enable_save must be a boolean")
        return cls(
            backend=_require_string(data, "backend", default="mooncake"),
            mooncake=MooncakeClientConfig.from_mapping(mooncake_data),
            model_revision=_require_string(data, "model_revision"),
            tokenizer_revision=_require_string(data, "tokenizer_revision"),
            min_tokens=_positive_int(data, "min_tokens", default=1024),
            load_timeout_ms=_positive_int(data, "load_timeout_ms", default=30_000),
            transfer_concurrency=_positive_int(data, "transfer_concurrency", default=2),
            failure_policy=_require_string(data, "failure_policy", default="cold_miss"),
            enable_save=enable_save,
        )
