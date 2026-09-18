# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""External test supervisor: never creates a new runtime in an old worker parent."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from pypto_serving.transfer.owner import _receive, _send
from pypto_serving.transfer.types import OwnerRef


def validate_retirement(record, *, run_id, role, generation, ranks, pid, returncode):
    expected = [asdict(OwnerRef(run_id, r, generation, generation, role)) for r in range(ranks)]
    other = "receiver" if role == "sender" else "sender"
    peers = [asdict(OwnerRef(run_id, r, generation, generation, other)) for r in range(ranks)]
    if (returncode != 75 or record is None
            or record.get("kind") != "worker_retirement_certificate"
            or record.get("run_id") != run_id or record.get("role") != role
            or record.get("generation") != generation or record.get("parent_pid") != pid
            or record.get("owners") != expected or record.get("peers") != peers):
        raise RuntimeError("old worker exit and complete paired-owner certificate required")


def execute_worker(args, generation, fault):
    command = [sys.executable, "-u", str(Path(__file__).with_name("production_pair.py")),
               "--worker-child", "--role", args.role, "--devices", args.devices,
               "--local-host", args.local_host, "--receiver-host", args.receiver_host,
               "--port", str(args.port), "--run-id", args.run_id,
               "--output-dir", str(args.output_dir / f"generation-{generation}"),
               "--generation", str(generation)]
    command += (["--peer-loss-inflight"] if args.peer_loss_inflight else ["--lost-reply"]) if fault else ["--reject-stale"]
    certificate = None
    succeeded = False
    expired = threading.Event()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, start_new_session=True) as process:
        def expire():
            expired.set()
            try:
                # This process group was created exclusively for this invocation.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        timer = threading.Timer(420, expire)
        timer.start()
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("kind") == "worker_retirement_certificate":
                    if certificate is not None:
                        raise RuntimeError("duplicate retirement certificate")
                    certificate = event
                if event.get("kind") == "phase_c_result":
                    succeeded = (event.get("run_id") == args.run_id
                                 and event.get("role") == args.role
                                 and event.get("generation") == generation
                                 and event.get("ranks") == len(args.devices.split(","))
                                 and event.get("status") == "ok"
                                 and event.get("payload_validated") is True)
            code = process.wait()
        except BaseException:
            expire()
            process.wait()
            raise
        finally:
            timer.cancel()
            timer.join()
    if expired.is_set():
        raise TimeoutError("worker budget expired; recovery remains blocked")
    if fault:
        validate_retirement(certificate, run_id=args.run_id, role=args.role,
                            generation=generation, ranks=len(args.devices.split(",")),
                            pid=process.pid, returncode=code)
    elif code != 0 or not succeeded:
        raise RuntimeError("fresh worker transfer did not pass")
    return process.pid


def supervise_pair(args):
    listener = None
    if args.role == "receiver":
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(180)
        listener.bind((args.local_host, args.port + 1))
        listener.listen(1)
        channel, _ = listener.accept()
    else:
        deadline = time.monotonic() + 120
        while True:
            try:
                channel = socket.create_connection((args.receiver_host, args.port + 1), timeout=10)
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                threading.Event().wait(0.1)
    try:
        channel.settimeout(480)

        def exchange(kind, **fields):
            _send(channel, dict(kind=kind, run_id=args.run_id, role=args.role,
                                generation=args.generation, **fields))
            peer = _receive(channel)
            other = "receiver" if args.role == "sender" else "sender"
            if (peer.get("kind") != kind or peer.get("run_id") != args.run_id
                    or peer.get("role") != other or peer.get("generation") != args.generation):
                raise RuntimeError("stale or mismatched supervisor message")
            return peer

        old_pid = execute_worker(args, args.generation, True)
        # A peer sends this only after wait() plus a complete owner-death certificate.
        exchange("worker_parent_reaped", parent_pid=old_pid)
        print(json.dumps(dict(kind="paired_parent_death_fence", run_id=args.run_id,
                              role=args.role, generation=args.generation)), flush=True)
        new_pid = execute_worker(args, args.generation + 1, False)
        exchange("fresh_transfer_validated", parent_pid=new_pid)
        print(json.dumps(dict(kind="paired_recovery_result", run_id=args.run_id,
                              role=args.role, generation=args.generation + 1,
                              old_parent_pid=old_pid, new_parent_pid=new_pid,
                              ranks=len(args.devices.split(",")), status="ok")), flush=True)
    finally:
        channel.close()
        if listener is not None:
            listener.close()
