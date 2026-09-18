#!/usr/bin/env python3
"""One-command, exact-PID Phase F orchestration for serving-a/b."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shlex
import subprocess
import time
from pathlib import Path


_IDENTITY = re.compile(r"^[A-Za-z0-9_.-]+$")


def _run(argv: list[str], *, stdin: str | None = None, timeout: float = 180) -> str:
    completed = subprocess.run(
        argv,
        input=stdin,
        text=True,
        check=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def _ssh(host: str, argv: list[str], **kwargs) -> str:
    # OpenSSH concatenates positional command arguments before the remote
    # shell parses them.  Quote once here so a nested ``bash -lc`` receives its
    # complete command as one argument instead of only the first word.
    return _run(["ssh", host, shlex.join(argv)], **kwargs)


def _docker(host: str, container: str, argv: list[str], **kwargs) -> str:
    return _ssh(host, ["docker", "exec", container, *argv], **kwargs)


def _copy_container_tree(
    host: str,
    container: str,
    source: str,
    destination: Path,
) -> None:
    """Copy one container directory without treating its path as a host path."""
    staging = f"/tmp/pypto-phase-f-evidence-{secrets.token_hex(12)}"
    try:
        _ssh(
            host,
            ["docker", "cp", f"{container}:{source}", staging],
            timeout=600,
        )
        _run(
            ["scp", "-r", f"{host}:{staging}", str(destination)],
            timeout=600,
        )
    finally:
        try:
            _ssh(host, ["rm", "-rf", "--", staging], timeout=30)
        except subprocess.SubprocessError:
            pass


def _shell_exports(values: dict[str, object]) -> str:
    return "; ".join(
        f"export {name}={shlex.quote(str(value))}" for name, value in values.items()
    )


def _wait_health(
    host: str,
    container: str,
    url: str,
    timeout: float,
    *,
    expected: dict[str, object],
) -> None:
    deadline = time.monotonic() + timeout
    last = "not started"
    while time.monotonic() < deadline:
        try:
            last = _docker(
                host,
                container,
                ["curl", "-fsS", "--max-time", "5", url],
                timeout=15,
            )
            payload = json.loads(last)
            if payload.get("status") == "ok" and all(
                payload.get(name) == value for name, value in expected.items()
            ):
                return
        except (subprocess.SubprocessError, ValueError) as exc:
            last = type(exc).__name__
        time.sleep(2)
    raise TimeoutError(f"health did not become ready: {url} ({last})")


def _launcher_pid(host: str, container: str, pid_file: str) -> int | None:
    command = (
        f"if test -f {shlex.quote(pid_file)}; then "
        f"cat {shlex.quote(pid_file)}; else printf __ABSENT__; fi"
    )
    value = _docker(host, container, ["bash", "-lc", command], timeout=15)
    if value == "__ABSENT__":
        return None
    if not value.isdigit() or int(value) < 2:
        raise RuntimeError(f"invalid launcher PID file: {pid_file}")
    return int(value)


def _child_pid_file(pid_file: str) -> str:
    if pid_file.endswith("/router-service.pid"):
        return pid_file[: -len("router-service.pid")] + "router-python.pid"
    if pid_file.endswith("/service.pid"):
        return pid_file[: -len("service.pid")] + "python.pid"
    if pid_file.endswith("/suite-service.pid"):
        return pid_file[: -len("suite-service.pid")] + "suite-python.pid"
    raise ValueError(f"unsupported launcher PID file: {pid_file}")


def _pid_alive(host: str, container: str, pid: int) -> bool:
    try:
        _docker(host, container, ["kill", "-0", str(pid)], timeout=15)
    except subprocess.CalledProcessError as exc:
        # ssh propagates the remote command's exit status.  Only the normal
        # `kill -0` false result proves absence; transport failure (normally
        # 255) is unknown and must be retried by the cleanup caller.
        if exc.returncode == 1:
            return False
        raise
    return True


def _signal_if_alive(
    host: str,
    container: str,
    pid: int | None,
    signal: str,
) -> bool:
    if pid is None or not _pid_alive(host, container, pid):
        return False
    _docker(host, container, ["kill", signal, str(pid)], timeout=15)
    return True


def _stop_launcher(host: str, container: str, pid_file: str) -> dict[str, object]:
    pid = _launcher_pid(host, container, pid_file)
    if pid is None:
        return {"pid_file": pid_file, "state": "absent"}
    child_file = _child_pid_file(pid_file)
    child_pid = _launcher_pid(host, container, child_file)
    if not _pid_alive(host, container, pid):
        return {
            "pid_file": pid_file,
            "pid": pid,
            "child_pid": child_pid,
            "state": "already-stopped",
        }

    # Bash defers a TERM trap while it is blocked in `wait child`.  Signal the
    # recorded service child first; the launcher then reaps it and executes its
    # session-owned cleanup path.  Never select processes by name.
    _signal_if_alive(host, container, child_pid, "-TERM")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _pid_alive(host, container, pid):
            return {
                "pid_file": pid_file,
                "pid": pid,
                "child_pid": child_pid,
                "state": "stopped",
                "forced": False,
            }
        time.sleep(2)

    # A live HTTP request can keep uvicorn in graceful shutdown indefinitely.
    # The exact child identity has already exhausted its TERM budget; killing
    # only that child lets the launcher finish its owned-session cleanup.
    _signal_if_alive(host, container, child_pid, "-KILL")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not _pid_alive(host, container, pid):
            return {
                "pid_file": pid_file,
                "pid": pid,
                "child_pid": child_pid,
                "state": "stopped",
                "forced": True,
            }
        time.sleep(2)
    return {
        "pid_file": pid_file,
        "pid": pid,
        "child_pid": child_pid,
        "state": "term-timeout",
        "forced": True,
    }


def _stop_with_retries(
    host: str,
    container: str,
    pid_file: str,
) -> dict[str, object]:
    last_error = ""
    for attempt in range(1, 4):
        try:
            return _stop_launcher(host, container, pid_file)
        except subprocess.SubprocessError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                time.sleep(2)
    return {
        "pid_file": pid_file,
        "state": "cleanup-error",
        "error": last_error,
    }


def _record_retained_launcher(
    host: str,
    container: str,
    pid_file: str,
) -> dict[str, object]:
    pid = _launcher_pid(host, container, pid_file)
    child_file = _child_pid_file(pid_file)
    child_pid = _launcher_pid(host, container, child_file)
    if pid is None or child_pid is None:
        return {
            "pid_file": pid_file,
            "state": "retain-error",
            "error": "launcher or child PID file is absent",
        }
    if not _pid_alive(host, container, pid) or not _pid_alive(
        host, container, child_pid
    ):
        return {
            "pid_file": pid_file,
            "pid": pid,
            "child_pid": child_pid,
            "state": "retain-error",
            "error": "launcher or child process is not alive",
        }
    return {
        "pid_file": pid_file,
        "pid": pid,
        "child_pid": child_pid,
        "state": "retained",
    }


def _record_retained_with_retries(
    host: str,
    container: str,
    pid_file: str,
) -> dict[str, object]:
    last_error = ""
    for attempt in range(1, 4):
        try:
            return _record_retained_launcher(host, container, pid_file)
        except subprocess.SubprocessError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                time.sleep(2)
    return {
        "pid_file": pid_file,
        "state": "retain-error",
        "error": last_error,
    }


def _write_pd_config(
    host: str,
    container: str,
    path: str,
    document: dict[str, object],
) -> None:
    body = json.dumps(document, indent=2, sort_keys=True) + "\n"
    command = f"tee {shlex.quote(path)} >/dev/null"
    _ssh(
        host,
        ["docker", "exec", "-i", container, "bash", "-lc", command],
        stdin=body,
        timeout=30,
    )


def _start(
    host: str,
    container: str,
    launcher: str,
    values: dict[str, object],
) -> None:
    command = (
        f"{_shell_exports(values)}; "
        f"exec {shlex.quote(launcher)}"
    )
    _ssh(
        host,
        ["docker", "exec", "-d", container, "setsid", "bash", "-lc", command],
        timeout=30,
    )


def _start_suite(
    host: str,
    container: str,
    repo: str,
    control_dir: str,
    driver: str,
    suite_args: list[str],
) -> None:
    command = (
        f"cd {shlex.quote(repo)}; "
        "exec tests/manual/pd/phase_f/run_remote_suite.sh "
        f"{shlex.quote(control_dir)} {shlex.quote(driver)} {shlex.join(suite_args)}"
    )
    _ssh(
        host,
        ["docker", "exec", "-d", container, "setsid", "bash", "-lc", command],
        timeout=30,
    )


def _wait_suite(
    host: str,
    container: str,
    control_dir: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    exit_file = f"{control_dir}/exit-code"
    last_transport_error = ""
    while time.monotonic() < deadline:
        command = (
            f"if test -f {shlex.quote(exit_file)}; then "
            f"cat {shlex.quote(exit_file)}; else printf __RUNNING__; fi"
        )
        try:
            status = _docker(
                host,
                container,
                ["bash", "-lc", command],
                timeout=15,
            )
        except subprocess.SubprocessError as exc:
            last_transport_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
            continue
        if status == "__RUNNING__":
            time.sleep(2)
            continue
        if status == "0":
            return
        tail = _docker(
            host,
            container,
            ["tail", "-n", "80", f"{control_dir}/stdout.log"],
            timeout=30,
        )
        raise RuntimeError(f"remote Phase F suite failed with exit {status}:\n{tail}")
    raise TimeoutError(
        "remote Phase F suite exceeded its timeout; "
        f"last transport error={last_transport_error or 'none'}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", default="/workspace/pypto-serving")
    parser.add_argument("--run-root", default="/workspace/phase-f-runs")
    parser.add_argument("--env-file", default="/home/sj/git/env_all.sh")
    parser.add_argument(
        "--model-dir", default="/models/dsv4-flash-0731-dspark-w8a8"
    )
    parser.add_argument("--container", default="openeuler-2403-DS")
    parser.add_argument("--prefill-ssh", default="serving-a-sj")
    parser.add_argument("--decode-ssh", default="serving-b-sj")
    # serving-a currently exposes its RoCE-facing address as 192.169.0.173
    # inside the host-networked test container.  Keep the default aligned with
    # the frozen Phase D/E topology so ADXL never tries to bind a stale address.
    parser.add_argument("--prefill-data-host", default="192.169.0.173")
    parser.add_argument("--decode-data-host", default="192.169.0.85")
    parser.add_argument("--control-port", type=int, default=29015)
    parser.add_argument("--prefill-port", type=int, default=8111)
    parser.add_argument("--decode-port", type=int, default=8112)
    parser.add_argument("--router-port", type=int, default=8110)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mode", choices=("suite", "soak"), default="suite")
    parser.add_argument("--cases-file", default="cases.json")
    parser.add_argument("--soak-requests", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--keep-services",
        action="store_true",
        help="retain healthy Router/P/D processes after a successful suite",
    )
    parser.add_argument("--startup-timeout-seconds", type=float, default=1800)
    parser.add_argument("--request-timeout-seconds", type=float, default=7200)
    parser.add_argument("--local-evidence-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not _IDENTITY.fullmatch(args.run_id):
        raise ValueError("run-id contains unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", args.cases_file):
        raise ValueError("cases-file must be a JSON basename inside phase_f")
    if args.soak_requests < 1:
        raise ValueError("soak-requests must be positive")
    for name, value in (
        ("repo", args.repo),
        ("run-root", args.run_root),
        ("env-file", args.env_file),
        ("model-dir", args.model_dir),
    ):
        path = Path(value)
        if not path.is_absolute() or path == Path("/"):
            raise ValueError(f"{name} must be an explicit absolute path")
    if not Path(args.run_root).name.startswith("phase-f-"):
        raise ValueError("run-root basename must start with phase-f-")
    if args.local_evidence_dir.exists():
        raise FileExistsError(args.local_evidence_dir)

    repo = args.repo.rstrip("/")
    run_root = args.run_root.rstrip("/")
    launcher = f"{repo}/tests/manual/pd/phase_f/run_pd_k7_node.sh"
    router_launcher = f"{repo}/tests/manual/pd/phase_f/run_router.sh"
    config_file = f"{run_root}/{args.run_id}.json"
    names = {
        "p": f"{args.run_id}-prefill",
        "d": f"{args.run_id}-decode",
        "router": f"{args.run_id}-router",
        "suite": f"{args.run_id}-suite",
        "suite_control": f"{args.run_id}-suite-control",
    }
    document = {
        "runtime": {
            "run_id": args.run_id,
            "prefill": [
                {
                    "host": args.prefill_data_host,
                    "port": args.prefill_port,
                    "node_id": f"{args.run_id}-p",
                    "control_host": args.prefill_data_host,
                    "control_port": args.control_port,
                    "transfer_hostname": args.prefill_data_host,
                }
            ],
            "decode": [
                {
                    "host": args.decode_data_host,
                    "port": args.decode_port,
                    "node_id": f"{args.run_id}-d",
                    "control_host": args.decode_data_host,
                    "control_port": args.control_port,
                    "transfer_hostname": args.decode_data_host,
                }
            ],
        },
        "observability": {"root": f"{run_root}/serving_log/pd_disag"},
    }
    stops = []
    result = "FAIL"
    try:
        for host in (args.prefill_ssh, args.decode_ssh):
            _docker(
                host,
                args.container,
                ["mkdir", "-p", run_root],
                timeout=30,
            )
            _write_pd_config(
                host,
                args.container,
                config_file,
                document,
            )
        common = {
            "PD_CONFIG": config_file,
            "PD_PROFILE": str(args.profile).lower(),
            "PD_EVIDENCE_ROOT": run_root,
            "PYPTO_STACK_ENV_FILE": args.env_file,
            "PYPTO_DSV4_DSPARK_MODEL_DIR": args.model_dir,
        }
        _start(
            args.decode_ssh,
            args.container,
            launcher,
            {
                **common,
                "PD_ROLE": "decode",
                "PD_NODE_ID": f"{args.run_id}-d",
                "PD_API_PORT": args.decode_port,
                "PD_EVIDENCE_NAME": names["d"],
            },
        )
        _start(
            args.prefill_ssh,
            args.container,
            launcher,
            {
                **common,
                "PD_ROLE": "prefill",
                "PD_NODE_ID": f"{args.run_id}-p",
                "PD_API_PORT": args.prefill_port,
                "PD_EVIDENCE_NAME": names["p"],
            },
        )
        _wait_health(
            args.decode_ssh,
            args.container,
            f"http://127.0.0.1:{args.decode_port}/health",
            args.startup_timeout_seconds,
            expected={
                "run_id": args.run_id,
                "role": "decode",
                "generation": 1,
                "control_incarnation": 1,
            },
        )
        _wait_health(
            args.prefill_ssh,
            args.container,
            f"http://127.0.0.1:{args.prefill_port}/health",
            args.startup_timeout_seconds,
            expected={
                "run_id": args.run_id,
                "role": "prefill",
                "generation": 1,
                "control_incarnation": 1,
            },
        )
        _start(
            args.prefill_ssh,
            args.container,
            router_launcher,
            {
                **common,
                "ROUTER_PORT": args.router_port,
                "ROUTER_EVIDENCE_NAME": names["router"],
            },
        )
        _wait_health(
            args.prefill_ssh,
            args.container,
            f"http://127.0.0.1:{args.router_port}/health",
            args.startup_timeout_seconds,
            expected={
                "run_id": args.run_id,
                "control_incarnation": 1,
            },
        )
        suite_args = [
            "--router-url",
            f"http://127.0.0.1:{args.router_port}",
            "--prefill-url",
            f"http://127.0.0.1:{args.prefill_port}",
            "--decode-url",
            f"http://{args.decode_data_host}:{args.decode_port}",
            "--evidence-dir",
            f"{run_root}/{names['suite']}",
            "--concurrency",
            str(args.concurrency),
        ]
        if args.mode == "suite":
            suite_args.extend(
                [
                    "--repeat",
                    str(args.repeat),
                    "--cases",
                    f"{repo}/tests/manual/pd/phase_f/{args.cases_file}",
                ]
            )
            if args.profile:
                suite_args.append("--profile")
            driver = "run_suite.py"
        else:
            if args.profile:
                raise ValueError("profile is supported only for suite mode")
            suite_args.extend(["--requests", str(args.soak_requests)])
            driver = "run_soak.py"
        _start_suite(
            args.prefill_ssh,
            args.container,
            repo,
            f"{run_root}/{names['suite_control']}",
            driver,
            suite_args,
        )
        _wait_suite(
            args.prefill_ssh,
            args.container,
            f"{run_root}/{names['suite_control']}",
            args.request_timeout_seconds,
        )
        result = "PASS"
    finally:
        stops.append(
            _stop_with_retries(
                args.prefill_ssh,
                args.container,
                f"{run_root}/{names['suite_control']}/suite-service.pid",
            )
        )
        retain_services = args.keep_services and result == "PASS"
        service_actions = (
            (args.prefill_ssh, f"{run_root}/{names['router']}/router-service.pid"),
            (args.prefill_ssh, f"{run_root}/{names['p']}/service.pid"),
            (args.decode_ssh, f"{run_root}/{names['d']}/service.pid"),
        )
        for host, pid_file in service_actions:
            action = (
                _record_retained_with_retries
                if retain_services
                else _stop_with_retries
            )
            stops.append(action(host, args.container, pid_file))
        args.local_evidence_dir.mkdir(parents=True, exist_ok=False)
        for label, host, name in (
            ("prefill", args.prefill_ssh, names["p"]),
            ("decode", args.decode_ssh, names["d"]),
            ("router", args.prefill_ssh, names["router"]),
            ("suite", args.prefill_ssh, names["suite"]),
            ("suite-control", args.prefill_ssh, names["suite_control"]),
        ):
            destination = args.local_evidence_dir / label
            try:
                _copy_container_tree(
                    host,
                    args.container,
                    f"{run_root}/{name}",
                    destination,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                (args.local_evidence_dir / f"{label}-copy-error.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
        (args.local_evidence_dir / "orchestration.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": args.run_id,
                    "result": result,
                    "services_retained": retain_services,
                    "stops": stops,
                    "finished_ns": time.time_ns(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    allowed_states = {"absent", "stopped", "already-stopped"}
    if args.keep_services:
        allowed_states.add("retained")
    return 0 if result == "PASS" and all(
        stop["state"] in allowed_states for stop in stops
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
