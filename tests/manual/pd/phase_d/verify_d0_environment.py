"""Verify the Phase D D0 source, toolchain, imports, and transfer overlay."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def _sha256(path: Path) -> str:
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _inside(path: str | os.PathLike[str], root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


def _module_file(module_name: str) -> str:
    module = importlib.import_module(module_name)
    path = getattr(module, "__file__", None)
    assert path, f"{module_name} has no import file"
    return str(Path(path).resolve())


stage = Path(__file__).resolve().parent
lock: dict[str, Any] = json.loads((stage / "d0-lock.json").read_text())
assert lock["schema"] == 1
assert str(stage) == lock["stage"], (stage, lock["stage"])

base_path = stage / "source-manifest.json"
assert _sha256(base_path) == lock["base_manifest_sha256"]
base = json.loads(base_path.read_text())
base_files: dict[str, str] = base["files"]
overlays: dict[str, str] = lock["overlay_files"]

for relative, old_digest in base_files.items():
    path = stage / relative
    expected = overlays.get(relative, old_digest)
    assert path.exists() or path.is_symlink(), relative
    assert _sha256(path) == expected, relative
for relative, expected in overlays.items():
    path = stage / relative
    assert path.exists() or path.is_symlink(), relative
    assert _sha256(path) == expected, relative

actual_repositories = {item["path"]: item["head"] for item in base["repositories"]}
assert actual_repositories == lock["repositories"]

for relative, expected in lock["root_files"].items():
    assert _sha256(stage / relative) == expected, relative

imports = {
    name: _module_file(name)
    for name in (
        "pypto",
        "pypto.pypto_core",
        "simpler",
        "simpler.chip_service",
        "simpler_setup",
        "pypto_serving",
        "golden",
        "ptoas",
        "ptodsl",
    )
}
for name, path in imports.items():
    assert _inside(path, stage), (name, path)

external_imports = {
    name: _module_file(name)
    for name in ("torch", "mooncake.engine", "safetensors", "fastapi", "uvicorn")
}
assert _inside(external_imports["mooncake.engine"], Path(os.environ["MLIR_PYTHON_ROOT"]))
for native_name in lock["mooncake_native_libraries"]:
    assert (Path(os.environ["MOONCAKE_LIB_DIR"]) / native_name).is_file(), native_name

from pypto.ir.distributed_compiled_program import (  # noqa: E402
    DistributedCompiledProgram,
)
from pypto.runtime import ensure_pto_isa_root  # noqa: E402
from pypto_serving.model.deepseek_dspark import (  # noqa: E402
    DeepSeekV4DSparkPyptoExecutor,
)
from pypto_serving.transfer.agent import TransferAgent  # noqa: E402

assert "chip_service_factories" in inspect.signature(
    DistributedCompiledProgram.prepare
).parameters
assert callable(DeepSeekV4DSparkPyptoExecutor)
assert callable(TransferAgent)

isa = ensure_pto_isa_root()
assert _inside(isa, stage)
pin = (stage / "pypto/runtime/pto_isa.pin").read_text().strip()
assert pin == lock["pto_isa"]
assert subprocess.check_output(
    ["git", "-C", str(isa), "rev-parse", "HEAD"], text=True
).strip() == pin

ptoas_binary = Path(os.environ["PTOAS_ROOT"]) / "ptoas"
version = subprocess.check_output([str(ptoas_binary), "--version"], text=True).strip()
assert version == lock["ptoas_version"]

native = stage / lock["simpler_native"]
assert native.is_file(), native
native_sha256 = _sha256(native)
assert native_sha256 == lock["simpler_native_sha256"]
caches = sorted((stage / "pypto/runtime/build/cache").rglob("CMakeCache.txt"))
assert caches
for cache in caches:
    expected_home = f"CMAKE_HOME_DIRECTORY:INTERNAL={stage}/pypto/runtime/"
    assert expected_home in cache.read_text(), cache

result = {
    "status": "ok",
    "lock_schema": lock["schema"],
    "repositories": actual_repositories,
    "imports": imports,
    "external_imports": external_imports,
    "base_files_verified": len(base_files),
    "overlay_files_verified": len(overlays),
    "root_files_verified": len(lock["root_files"]),
    "ptoas": version,
    "isa": pin,
    "native": str(native),
    "native_sha256": native_sha256,
    "npu_test": "not run",
}
(stage / "d0-verified.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
