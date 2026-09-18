"""Compatibility entrypoint for the Phase D D0 verifier."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().with_name("verify_d0_environment.py")),
    run_name="__main__",
)
