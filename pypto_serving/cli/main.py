# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parse, resolve, and dispatch the PyPTO Serving command."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator, Sequence

from pypto_serving.cli.parser import parse_launch_options
from pypto_serving.config.resolve import resolve_launch_options


_LEGACY_SERVING_PROFILE_ENV = ("SA_PROFILE_OUTPUT", "SA_PROFILE_LEVEL")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, resolve, and dispatch serving or offline generation."""
    options = parse_launch_options(argv)
    _warn_deprecated_serving_profile_env(options.profile.enabled)
    try:
        with _startup_log_context(enabled=not options.show_startup_logs):
            launch = resolve_launch_options(options)
    except ValueError as exc:
        print(f"pypto-serving: error: {exc}", file=sys.stderr)
        return 2

    if launch.prompts is not None:
        from pypto_serving.cli.generate import generate

        generate(launch.engine, launch.prompts, launch.generation)
    else:
        from pypto_serving.cli.serve import serve

        serve(launch.engine, launch.generation, host=launch.serve_host, port=launch.serve_port)
    return 0


def _warn_deprecated_serving_profile_env(profile_enabled: bool) -> None:
    """Report ignored legacy profiling variables outside profile mode."""
    if profile_enabled:
        return
    legacy_vars = [name for name in _LEGACY_SERVING_PROFILE_ENV if name in os.environ]
    if legacy_vars:
        print(
            "WARNING: "
            f"{', '.join(legacy_vars)} are deprecated for HTTP serving and are ignored "
            "without --profile. Use --profile with --profile-output/--profile-level instead.",
            file=sys.stderr,
            flush=True,
        )


@contextlib.contextmanager
def _startup_log_context(*, enabled: bool) -> Iterator[None]:
    """Temporarily hide startup output and restore streams and log settings."""
    if not enabled:
        yield
        return
    old_log_level = os.environ.get("PTO_LOG_LEVEL")
    os.environ.setdefault("PTO_LOG_LEVEL", "error")
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        if old_log_level is None:
            os.environ.pop("PTO_LOG_LEVEL", None)
        else:
            os.environ["PTO_LOG_LEVEL"] = old_log_level


if __name__ == "__main__":
    raise SystemExit(main())
