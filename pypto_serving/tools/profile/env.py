# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_OUTPUT = "./profile_out"
_DEFAULT_LEVEL = "e2e,kernel"
_VALID_LEVELS = frozenset({"e2e", "kernel", "verbose"})


@dataclass(frozen=True)
class ProfileConfig:
    """Environment-derived profiler configuration."""

    enabled: bool
    output: Path
    trace_file: Path
    fragments_dir: Path
    levels: frozenset[str]
    chip_swimlane_level: int = 0

    def includes(self, level: str) -> bool:
        return "verbose" in self.levels or level in self.levels

    @property
    def chip_artifacts_dir(self) -> Path:
        if self.output.suffix == ".json":
            return self.output.with_suffix(".chip")
        return self.output / "chip"


def load_profile_config(env: dict[str, str] | None = None) -> ProfileConfig:
    """Load SA_PROFILE_* settings.

    Profiling is enabled when either SA_PROFILE_OUTPUT or SA_PROFILE_LEVEL is
    present. When enabled without SA_PROFILE_OUTPUT, artifacts go to
    ./profile_out.
    """
    source = os.environ if env is None else env
    enabled = "SA_PROFILE_OUTPUT" in source or "SA_PROFILE_LEVEL" in source
    return create_profile_config(
        enabled=enabled,
        output=source.get("SA_PROFILE_OUTPUT", _DEFAULT_OUTPUT),
        levels=source.get("SA_PROFILE_LEVEL", _DEFAULT_LEVEL),
    )


def create_profile_config(
    *,
    enabled: bool,
    output: str | Path = _DEFAULT_OUTPUT,
    levels: str = _DEFAULT_LEVEL,
    chip_swimlane_level: int = 0,
) -> ProfileConfig:
    """Build an explicit profiler configuration."""
    output = Path(output).expanduser()
    parsed_levels = _parse_levels(levels)
    if (
        isinstance(chip_swimlane_level, bool)
        or not isinstance(chip_swimlane_level, int)
        or not 0 <= chip_swimlane_level <= 4
    ):
        raise ValueError("chip_swimlane_level must be an integer from 0 to 4")
    if chip_swimlane_level and not enabled:
        raise ValueError("chip_swimlane_level requires profiling to be enabled")
    trace_file, fragments_dir = _resolve_output(output)
    return ProfileConfig(
        enabled=enabled,
        output=output,
        trace_file=trace_file,
        fragments_dir=fragments_dir,
        levels=parsed_levels,
        chip_swimlane_level=chip_swimlane_level,
    )


def _parse_levels(raw: str) -> frozenset[str]:
    parts = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not parts:
        parts = set(_DEFAULT_LEVEL.split(","))
    unknown = parts - _VALID_LEVELS
    if unknown:
        raise ValueError(
            "SA_PROFILE_LEVEL contains unsupported value(s): "
            f"{', '.join(sorted(unknown))}. Supported: {', '.join(sorted(_VALID_LEVELS))}."
        )
    if "verbose" in parts:
        parts.update({"e2e", "kernel"})
    return frozenset(parts)


def _resolve_output(output: Path) -> tuple[Path, Path]:
    if output.suffix == ".json":
        trace_file = output
        fragments_dir = output.with_suffix(output.suffix + ".fragments")
    else:
        trace_file = output / "trace.json"
        fragments_dir = output / "fragments"
    return trace_file, fragments_dir
