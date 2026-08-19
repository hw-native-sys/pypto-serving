# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lazy, name-addressed safetensors access shared by every model family.

A checkpoint is read one tensor at a time, grouped by shard file, so staging a
single layer never materializes its neighbours. The family-specific parts — which
names exist, how they map to kernel weights, how they shard across ranks — belong
above this layer; all that lives here is the index and the reads.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import ContextManager, Protocol

import torch

logger = logging.getLogger(__name__)


class SafeTensorReader(Protocol):
    """Minimal safetensors reader protocol used by the lazy weight store."""

    def get_tensor(self, name: str) -> torch.Tensor:
        """Return one tensor by name."""
        raise NotImplementedError


class SafeOpenFn(Protocol):
    """Callable shape for injectable safetensors openers."""

    def __call__(self, path: Path, device: str) -> ContextManager[SafeTensorReader]:
        """Open one safetensors shard."""
        raise NotImplementedError


def default_safe_open(path: Path, device: str) -> ContextManager[SafeTensorReader]:
    """Open a safetensors shard without loading unrelated tensors."""
    try:
        from safetensors import safe_open  # noqa: PLC0415 -- optional dependency, resolved per call
    except ImportError as exc:
        raise RuntimeError("safetensors is required to read model weights.") from exc

    return safe_open(str(path), framework="pt", device=device)


class LazySafetensorsStore:
    """Name-addressed reader over a Hugging Face safetensors index.

    Subclasses add the family contract (which names must exist, how they are packed);
    they do not reimplement the reads. The three ``*_error`` templates are class
    attributes rather than hard-coded strings so a family can keep the diagnostics its
    users already recognise — the wording is part of that contract too.
    """

    missing_name_error = "Missing weight tensor in index: {name}"
    missing_names_error = "Checkpoint is missing required tensors: {names}"
    missing_shard_error = "Missing safetensors shard for weight load: {path}"

    def __init__(
        self,
        *,
        model_dir: str | Path,
        weight_map: Mapping[str, str],
        device: str = "cpu",
        safe_open_fn: SafeOpenFn | None = None,
    ) -> None:
        """Create a store from the Hugging Face safetensors index."""
        self.model_dir = Path(model_dir)
        self.weight_map = dict(weight_map)
        self.device = device
        # Resolved here, not at read time, and through a method so a family can point at
        # its own module-level opener — which is what test monkeypatching replaces.
        self._safe_open_fn = self._default_open_fn() if safe_open_fn is None else safe_open_fn

    def _default_open_fn(self) -> SafeOpenFn:
        """Return the opener used when the caller injects none."""
        return default_safe_open

    def __contains__(self, name: object) -> bool:
        """Return whether the checkpoint index exposes ``name``."""
        return isinstance(name, str) and name in self.weight_map

    def filename_for(self, name: str) -> str:
        """Return the safetensors shard filename for ``name``."""
        try:
            return self.weight_map[name]
        except KeyError as exc:
            raise KeyError(self.missing_name_error.format(name=name)) from exc

    def path_for(self, name: str) -> Path:
        """Return the shard path containing ``name``."""
        return self.model_dir / self.filename_for(name)

    def require(self, names: Iterable[str]) -> None:
        """Validate that all tensor names are present in the checkpoint index."""
        missing = [name for name in names if name not in self.weight_map]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
            raise KeyError(self.missing_names_error.format(names=f"{preview}{suffix}"))

    def load_tensor(self, name: str) -> torch.Tensor:
        """Load one tensor by name, leaving all unrelated shard tensors untouched."""
        return self.load_many([name])[name]

    def load_many(self, names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Load a set of named tensors grouped by shard file.

        Grouping is what keeps the read count at one open per shard rather than one per
        tensor, and the result is returned in the caller's order, de-duplicated.
        """
        unique_names = tuple(dict.fromkeys(names))
        self.require(unique_names)

        groups: dict[str, list[str]] = {}
        for name in unique_names:
            groups.setdefault(self.filename_for(name), []).append(name)

        loaded: dict[str, torch.Tensor] = {}
        for filename, shard_names in groups.items():
            path = self.model_dir / filename
            if not path.exists():
                raise FileNotFoundError(self.missing_shard_error.format(path=path))
            with self._safe_open_fn(path, self.device) as reader:
                for name in shard_names:
                    loaded[name] = reader.get_tensor(name)

        return {name: loaded[name] for name in unique_names}
