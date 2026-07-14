# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DeepSeek V4 HTTP generation accuracy guard for CI."""

from __future__ import annotations

import contextlib
import ctypes
import io
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dsv4-flash-w8a8"
PROMPT = "Huawei is"
MAX_NEW_TOKENS = 6
EXPECTED_TEXT = " a leading global provider of ICT"

STARTUP_TIMEOUT_SECONDS = 600
OVERALL_TIMEOUT_SECONDS = 1650
HEARTBEAT_SECONDS = 30

# How long to let the server group settle after SIGTERM before escalating to SIGKILL.
#
# On the normal path (test passed or failed on its own) we can afford to be patient.
#
# On the interrupted path we cannot. task-submit's watchdog sends SIGTERM to our
# process group and escalates to SIGKILL 3 seconds later; the --kill path allows
# 5 (KILL_GRACE). Once that SIGKILL lands, pytest is gone and nobody is left to
# reap the server — which is exactly how ~800 GiB of orphaned engines accumulated.
# So when we are being terminated, the whole teardown has to fit inside ~3s.
NORMAL_TERM_GRACE_SECONDS = 20.0
SIGNAL_TERM_GRACE_SECONDS = 1.0

# prctl(2) option number. Not exposed by the signal/os modules, so spell it out.
_PR_SET_PDEATHSIG = 1

_TERMINATION_SIGNALS = tuple(
    sig
    for sig in (
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGHUP", None),
    )
    if sig is not None
)


class TerminatedBySignal(KeyboardInterrupt):
    """Raised in the main thread when the CI runner asks this process to stop.

    Subclasses KeyboardInterrupt so pytest treats it as an interrupt and tears the
    session down, rather than recording it as an ordinary test failure and moving
    on to the next test.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(f"terminated by signal {signum}")
        self.signum = signum


@contextlib.contextmanager
def _raise_on_termination():
    """Turn SIGTERM/SIGINT/SIGHUP into an exception so that `finally` blocks run.

    CPython leaves SIGTERM at SIG_DFL, so a plain `kill` tears the interpreter down
    where it stands: no stack unwind, no `finally`, no `atexit`. The server we
    started with start_new_session=True is in its own session and process group, so
    it survives the signal that killed us and is inherited by init — still holding
    its NPU cards and several hundred GiB of shared memory. Raising from the handler
    is what gives the teardown below a chance to run at all.
    """
    if threading.current_thread() is not threading.main_thread():
        # signal.signal() is main-thread-only. Under pytest we always are on it;
        # degrade to a no-op rather than exploding if that ever stops being true.
        yield
        return

    def handler(signum: int, _frame) -> None:
        # Deafen ourselves before unwinding. The runner sends SIGTERM and then
        # escalates, and a second delivery landing inside the teardown would abort
        # it halfway — leaving behind exactly the orphan we are trying to prevent.
        # SIGKILL still gets through; PR_SET_PDEATHSIG below is the answer to that.
        for sig in _TERMINATION_SIGNALS:
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, signal.SIG_IGN)
        raise TerminatedBySignal(signum)

    previous: dict[int, object] = {}
    for sig in _TERMINATION_SIGNALS:
        try:
            previous[sig] = signal.signal(sig, handler)
        except (OSError, ValueError):
            continue
    try:
        yield
    finally:
        for sig, old_handler in previous.items():
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, old_handler)


def _set_pdeathsig() -> None:
    """Child-side preexec hook: ask the kernel to SIGKILL us if our parent dies.

    Backstop for the one signal we cannot catch. If pytest is SIGKILLed, no handler
    runs and no teardown happens — but PR_SET_PDEATHSIG is enforced by the kernel,
    so the server still dies. It only covers the process we fork directly, so the
    server's own workers can in principle linger; treat this as a safety net under
    _raise_on_termination(), not as a replacement for it.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    # Close the fork/prctl race: if the parent already died in that window, the
    # death signal fired before we asked for it and will never be re-sent.
    if os.getppid() == 1:
        os._exit(1)


def _task_devices() -> tuple[int, ...]:
    raw_devices = os.environ.get("TASK_DEVICE", "")
    try:
        devices = tuple(int(value.strip()) for value in raw_devices.split(",") if value.strip())
    except ValueError:
        pytest.fail(f"TASK_DEVICE must contain comma-separated integer device IDs, got {raw_devices!r}")
    if len(devices) != 8 or len(set(devices)) != 8 or any(device < 0 for device in devices):
        pytest.fail(f"TASK_DEVICE must contain exactly 8 unique non-negative device IDs, got {raw_devices!r}")
    return devices


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(model_dir: Path, devices: tuple[int, ...], port: int) -> list[str]:
    # Keep these serving options aligned with docs/dev/model/deepseek-v4.md.
    # CI substitutes only the checkpoint, task-submit devices, and free port.
    return [
        sys.executable,
        "python/cli/main.py",
        "--model",
        str(model_dir),
        "--served-model-name",
        MODEL_ID,
        "--backend",
        "npu",
        "--platform",
        "a2a3",
        "--devices",
        ",".join(str(device) for device in devices),
        "--dp",
        "1",
        "--tp",
        "8",
        "--block-size",
        "128",
        "--max-model-len",
        "260",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "512",
        "--long-prefill-token-threshold",
        "2048",
        "--no-enable-prefix-caching",
        "--port",
        str(port),
        "--show-startup-logs",
    ]


def _wait_for_health(process: subprocess.Popen, port: int, deadline: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    startup_deadline = min(deadline, time.monotonic() + STARTUP_TIMEOUT_SECONDS)
    next_heartbeat = time.monotonic()
    last_error: BaseException | None = None

    while time.monotonic() < startup_deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"DeepSeek server exited before becoming healthy (code={return_code})")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read())
            if response.status == 200 and payload == {"status": "ok"}:
                print("DeepSeek server is healthy", flush=True)
                return
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as exc:
            last_error = exc

        now = time.monotonic()
        if now >= next_heartbeat:
            print("Waiting for DeepSeek server startup...", flush=True)
            next_heartbeat = now + HEARTBEAT_SECONDS
        time.sleep(2)

    raise TimeoutError(f"DeepSeek server did not become healthy: {last_error}")


def _request_completion(process: subprocess.Popen, port: int, deadline: float) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(
            {
                "model": MODEL_ID,
                "prompt": PROMPT,
                "max_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "top_p": 1.0,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    results: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def send_request() -> None:
        try:
            timeout = max(1.0, deadline - time.monotonic())
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                results.put((True, json.loads(body)))
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = "<failed to read error body>"
            results.put(
                (False, RuntimeError(f"completion request returned HTTP {exc.code}: {error_body}"))
            )
        except BaseException as exc:
            results.put((False, exc))

    threading.Thread(target=send_request, name="deepseek-completion", daemon=True).start()
    while time.monotonic() < deadline:
        try:
            succeeded, value = results.get(timeout=HEARTBEAT_SECONDS)
        except queue.Empty:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"DeepSeek server exited during generation (code={return_code})"
                ) from None
            print("Waiting for DeepSeek completion...", flush=True)
            continue
        if succeeded:
            if not isinstance(value, dict):
                raise TypeError(f"completion response must be a JSON object, got {type(value).__name__}")
            return value
        if isinstance(value, BaseException):
            raise value
        raise RuntimeError(f"completion request failed: {value}")
    raise TimeoutError("DeepSeek completion exceeded the end-to-end timeout")


def _stop_process_group(
    process: subprocess.Popen, term_grace: float = NORMAL_TERM_GRACE_SECONDS
) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        print(f"WARNING: failed to terminate process group {process.pid}: {exc}", flush=True)
        return

    try:
        process.wait(timeout=term_grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"WARNING: process group {process.pid} still alive after SIGKILL", flush=True)
        except Exception as exc:
            print(f"WARNING: failed to reap process group {process.pid}: {exc}", flush=True)
        return
    except Exception as exc:
        print(f"WARNING: failed to wait for process group {process.pid}: {exc}", flush=True)
        return

    # The server parent may exit before a worker child. Give the process group a
    # short grace period, then kill any remaining descendants. Bounded by term_grace
    # so the interrupted path still fits inside the runner's SIGKILL window.
    shutdown_deadline = time.monotonic() + min(2.0, term_grace)
    while time.monotonic() < shutdown_deadline:
        try:
            os.killpg(process.pid, 0)
        except OSError:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass


def _print_server_log(log_path: Path) -> None:
    if not log_path.exists():
        return
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - 50000))
            content = log_file.read().decode("utf-8", errors="replace")
    except OSError as exc:
        print(f"WARNING: failed to read DeepSeek server log: {exc}", flush=True)
        return
    print("\n--- DeepSeek server log (tail) ---", flush=True)
    print(content, flush=True)


def test_deepseek_v4_http_completion_matches_expected_text(tmp_path: Path) -> None:
    model_dir_env = os.environ.get("PYPTO_DSV4_MODEL_DIR")
    model_dir = Path(model_dir_env) if model_dir_env else None
    if model_dir is None or not model_dir.is_dir():
        pytest.fail(f"PYPTO_DSV4_MODEL_DIR not set or not a directory: {model_dir}")
    devices = _task_devices()
    port = _unused_local_port()
    log_path = tmp_path / "deepseek-v4-server.log"
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS

    with _raise_on_termination():
        try:
            with log_path.open("w", encoding="utf-8") as server_log:
                # `process` is bound before the try so that a signal arriving between
                # fork and assignment still reaches the teardown. Popen's own internals
                # are covered by PR_SET_PDEATHSIG.
                process = None
                try:
                    process = subprocess.Popen(
                        _server_command(model_dir, devices, port),
                        cwd=ROOT,
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        preexec_fn=_set_pdeathsig,
                        text=True,
                    )
                    _wait_for_health(process, port, deadline)
                    response = _request_completion(process, port, deadline)
                    print(f"DeepSeek completion response: {response}", flush=True)

                    assert response.get("model") == MODEL_ID
                    choices = response.get("choices")
                    assert isinstance(choices, list) and len(choices) == 1
                    assert choices[0].get("text") == EXPECTED_TEXT
                    assert choices[0].get("finish_reason") == "length"
                finally:
                    if process is not None:
                        # Being torn down by the runner means we have ~3s before SIGKILL,
                        # so escalate to SIGKILL fast. On the normal path, be patient and
                        # let the engine shut down cleanly.
                        interrupted = isinstance(sys.exc_info()[1], TerminatedBySignal)
                        _stop_process_group(
                            process,
                            term_grace=(
                                SIGNAL_TERM_GRACE_SECONDS if interrupted else NORMAL_TERM_GRACE_SECONDS
                            ),
                        )
        except BaseException:
            _print_server_log(log_path)
            raise


def test_completion_http_error_includes_response_body(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "http://127.0.0.1/completions",
        500,
        "Internal Server Error",
        {},
        io.BytesIO(b"device allocation failed"),
    )

    def raise_http_error(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)

    class RunningProcess:
        @staticmethod
        def poll():
            return None

    with pytest.raises(RuntimeError, match="HTTP 500: device allocation failed"):
        _request_completion(RunningProcess(), 1, time.monotonic() + 1)


def test_stop_process_group_suppresses_final_wait_timeout(monkeypatch, capsys) -> None:
    class StuckProcess:
        pid = 123

        @staticmethod
        def wait(timeout):
            raise subprocess.TimeoutExpired("server", timeout)

    monkeypatch.setattr(os, "killpg", lambda *_args: None)

    _stop_process_group(StuckProcess())

    assert "still alive after SIGKILL" in capsys.readouterr().out


def test_stop_process_group_honours_term_grace(monkeypatch) -> None:
    """The interrupted path must escalate to SIGKILL well inside the runner's window."""
    waits: list[float] = []
    signals: list[int] = []

    class StuckProcess:
        pid = 123

        @staticmethod
        def wait(timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("server", timeout)

    monkeypatch.setattr(os, "killpg", lambda _pid, sig: signals.append(sig))

    _stop_process_group(StuckProcess(), term_grace=SIGNAL_TERM_GRACE_SECONDS)

    assert waits[0] == SIGNAL_TERM_GRACE_SECONDS
    assert signals[:2] == [signal.SIGTERM, signal.SIGKILL]


def test_sigterm_unwinds_the_stack_so_teardown_runs() -> None:
    """The whole point: a bare SIGTERM must not skip `finally`.

    Without _raise_on_termination() this test does not fail — it takes the pytest
    process down with it, which is precisely the production bug.
    """
    torn_down: list[str] = []

    with pytest.raises(TerminatedBySignal) as excinfo:
        with _raise_on_termination():
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                # The handler runs at the next bytecode boundary; give it one.
                time.sleep(1)
            finally:
                torn_down.append("server stopped")

    assert excinfo.value.signum == signal.SIGTERM
    assert torn_down == ["server stopped"]


def test_repeated_sigterm_cannot_abort_teardown() -> None:
    """The runner escalates. A second SIGTERM must not interrupt cleanup in progress."""
    torn_down: list[str] = []

    with pytest.raises(TerminatedBySignal):
        with _raise_on_termination():
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(1)
            finally:
                # Simulates the escalation landing mid-teardown. It must be ignored,
                # not turned into a second exception that abandons the cleanup.
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.2)
                torn_down.append("server stopped")

    assert torn_down == ["server stopped"]


def test_signal_handlers_are_restored_on_exit() -> None:
    original = signal.getsignal(signal.SIGTERM)
    with _raise_on_termination():
        assert signal.getsignal(signal.SIGTERM) is not original
    assert signal.getsignal(signal.SIGTERM) is original


def test_print_server_log_reads_only_tail(tmp_path, capsys) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_bytes(b"excluded-prefix\n" + b"x" * 60000 + b"\nincluded-tail\n")

    _print_server_log(log_path)

    output = capsys.readouterr().out
    assert "excluded-prefix" not in output
    assert "included-tail" in output
