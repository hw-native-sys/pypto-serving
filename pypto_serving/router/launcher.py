# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Starting and stopping serving replicas, locally or over ssh.

A replica takes minutes to become useful -- roughly 130 s for Qwen3-14B with a
warm kernel cache. Two consequences shape this module:

* A slot is reserved **before** anything is spawned. If two callers could pick
  the same device while the first launch was still loading, a fleet would eat a
  cluster's cards before the first replica answered.
* A launch that never reports ready is stopped and its slot returned. Otherwise
  a failed start holds a device for as long as the router runs.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from dataclasses import dataclass, field

from pypto_serving.router.config import HostSpec, ReplicaSpec, RouterConfig

logger = logging.getLogger(__name__)

# Long enough to notice a wedged transport, short enough that a stuck stop does
# not hold the shutdown path open.
COMMAND_TIMEOUT_SECONDS = 120.0


class LaunchError(RuntimeError):
    """A replica could not be started or stopped."""


class PoolExhausted(RuntimeError):
    """Every declared device already carries a replica."""


@dataclass(frozen=True)
class Slot:
    """One launchable device group on a host, and the replica it would become.

    A group, not a single card: one replica of DeepSeek V4 needs eight of them
    and one of Qwen3-14B needs one, so the group is the unit that can be
    allocated and freed.
    """

    host: HostSpec
    devices: tuple[int, ...]

    @property
    def name(self) -> str:
        # Named after the group's first device, which is also what fixes its
        # port, so a slot's identity survives a restart.
        return f"{self.host.name}-d{self.devices[0]}"

    @property
    def device(self) -> int:
        """The group's first device; what single-device templates expand to."""
        return self.devices[0]

    @property
    def device_list(self) -> str:
        return ",".join(str(device) for device in self.devices)

    @property
    def port(self) -> int:
        return self.host.port_for(self.devices)

    def replica_spec(self) -> ReplicaSpec:
        return ReplicaSpec(name=self.name, host=self.host.address(), port=self.port)

    @property
    def log_path(self) -> str:
        return f"{self.host.log_dir.rstrip('/')}/{self.name}.log"


def _expand(values, **substitutions: object) -> list[str]:
    """Substitute ``{device}``, ``{devices}`` and ``{port}`` in a template."""
    return [str(value).format(**substitutions) for value in values]


class Transport:
    """How a command is run on one host."""

    async def run(self, host: HostSpec, argv: list[str], env: dict[str, str],
                  *, detach: bool = False, log_path: str | None = None) -> None:
        """Run a command. ``detach`` starts something meant to outlive the call."""
        raise NotImplementedError


def _env_prefix(env: dict[str, str]) -> list[str]:
    """Render the environment as an ``env KEY=VALUE ...`` prefix.

    The remote side is reached through a command, not a login shell, so there is
    nowhere to export variables first.
    """
    if not env:
        return []
    return ["env", *(f"{key}={value}" for key, value in sorted(env.items()))]


async def _run_and_wait(argv: list[str]) -> str:
    """Run argv to completion with no shell, and return its output."""
    logger.info("running: %s", " ".join(shlex.quote(part) for part in argv))
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        raise LaunchError(f"could not run {argv[0]!r}: {exc}") from exc

    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(), timeout=COMMAND_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        process.kill()
        raise LaunchError(
            f"{argv[0]!r} did not return within {COMMAND_TIMEOUT_SECONDS:g}s"
        ) from exc

    output = (stdout or b"").decode(errors="replace").strip()
    if process.returncode != 0:
        raise LaunchError(f"{argv[0]!r} exited {process.returncode}: {output[-800:]}")
    return output


class LocalTransport(Transport):
    """Run the command on the machine the router is on.

    A launch is detached and never waited on: the serving process runs for the
    life of the deployment, and a model load takes minutes. ``start_new_session``
    keeps it alive if the router dies, which is the documented behaviour -- a
    crash must not cost a model load.
    """

    async def run(self, host: HostSpec, argv: list[str], env: dict[str, str],
                  *, detach: bool = False, log_path: str | None = None) -> None:
        command = [*_env_prefix(env), *argv]
        if not detach:
            await _run_and_wait(command)
            return

        logger.info("launching: %s", " ".join(shlex.quote(part) for part in command))
        try:
            if log_path:
                # The remote path does this with mkdir -p; the local one has to
                # do it here, or the first launch fails on a missing directory.
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                log = open(log_path, "ab")
            else:
                log = asyncio.subprocess.DEVNULL
        except OSError as exc:
            raise LaunchError(f"cannot open launch log {log_path}: {exc}") from exc
        try:
            await asyncio.create_subprocess_exec(
                *command,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=host.workdir,
                start_new_session=True,
            )
        except OSError as exc:
            raise LaunchError(f"could not start {command[0]!r}: {exc}") from exc
        finally:
            if log is not asyncio.subprocess.DEVNULL:
                log.close()


class SshTransport(Transport):
    """Run the command on another machine.

    ``BatchMode=yes`` matters: without it a missing or rejected key turns into a
    password prompt on a process with no terminal, and the launch hangs instead
    of failing.

    A launch is backgrounded on the remote side with its output redirected to a
    log, so ssh returns as soon as the process is started rather than holding
    the connection open for the minutes a model load takes. ssh's own exit
    status still reports the failures that matter here -- an unreachable host, a
    rejected key -- while whether the model loaded is answered later by the
    health probe, and diagnosed from that log.
    """

    def __init__(self, *, connect_timeout: float = 10.0) -> None:
        self._connect_timeout = connect_timeout

    def ssh_argv(self, host: HostSpec) -> list[str]:
        argv = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={int(self._connect_timeout)}",
        ]
        if host.identity_file:
            argv += ["-i", host.identity_file, "-o", "IdentitiesOnly=yes"]
        argv.append(str(host.ssh))
        return argv

    async def run(self, host: HostSpec, argv: list[str], env: dict[str, str],
                  *, detach: bool = False, log_path: str | None = None) -> None:
        if not host.ssh:
            raise LaunchError(f"host {host.name!r} has no ssh destination")
        # The remote side is a shell, so the payload is quoted into one string.
        payload = " ".join(shlex.quote(part) for part in [*_env_prefix(env), *argv])
        if host.workdir:
            payload = f"cd {shlex.quote(host.workdir)} && {payload}"
        if detach:
            target = shlex.quote(log_path) if log_path else "/dev/null"
            payload = f"mkdir -p $(dirname {target}) 2>/dev/null; nohup {payload} >> {target} 2>&1 &"
        await _run_and_wait([*self.ssh_argv(host), payload])


def transport_for(host: HostSpec) -> Transport:
    return LocalTransport() if host.is_local else SshTransport()


@dataclass
class LaunchedReplica:
    """A replica this router started, and therefore owns."""

    slot: Slot
    spec: ReplicaSpec


class HostPool:
    """Tracks which declared devices are free, and starts replicas on them.

    Allocation is synchronous and separate from launching precisely so that two
    concurrent callers cannot be handed the same device while the first launch
    is still loading its model.
    """

    def __init__(
        self,
        config: RouterConfig,
        *,
        transport_factory=transport_for,
    ) -> None:
        self._config = config
        self._transport_factory = transport_factory
        # Config order, local hosts first: fill the machine we are on before
        # reaching for another one.
        ordered = sorted(config.hosts, key=lambda host: not host.is_local)
        self._slots: list[Slot] = [
            Slot(host=host, devices=group)
            for host in ordered
            for group in host.device_groups()
        ]
        self._taken: dict[str, Slot] = {}

    @property
    def ceiling(self) -> int:
        return self._config.pool_ceiling

    @property
    def in_use(self) -> int:
        return len(self._taken)

    def free_slots(self) -> list[Slot]:
        return [slot for slot in self._slots if slot.name not in self._taken]

    def allocate(self) -> Slot:
        """Reserve the next free slot, or refuse.

        Synchronous on purpose: it must be impossible for an await to land
        between choosing a device and marking it taken.
        """
        if self.in_use >= self.ceiling:
            raise PoolExhausted(
                f"all {self.ceiling} declared device slots are in use"
            )
        for slot in self._slots:
            if slot.name not in self._taken:
                self._taken[slot.name] = slot
                return slot
        raise PoolExhausted("no free device slot")

    def log_path_for(self, name: str) -> str | None:
        """Where a slot's startup output went, for an error that needs to cite it."""
        slot = self._taken.get(name)
        return slot.log_path if slot else None

    def reserve(self, name: str) -> Slot | None:
        """Re-take a named slot, for adopting a replica after a restart."""
        if name in self._taken:
            return None
        for slot in self._slots:
            if slot.name == name:
                self._taken[name] = slot
                return slot
        return None

    def release(self, name: str) -> None:
        self._taken.pop(name, None)

    def serve_argv(self, slot: Slot) -> list[str]:
        """The pypto-serving command line for this slot."""
        host = slot.host
        argv = [
            host.python, "-m", "pypto_serving.cli",
            "--model", host.model,
            # --devices takes the whole group; serving derives the parallel
            # placement from it, and validates it against what the model's
            # kernels require.
            "--devices", slot.device_list,
            "--host", "0.0.0.0",
            "--port", str(slot.port),
            # Without this, startup logs go to /dev/null and a launch that fails
            # to load the model leaves nothing to diagnose it with.
            "--show-startup-logs",
        ]
        if host.served_model_name:
            argv += ["--served-model-name", host.served_model_name]
        argv += list(host.serve_args)
        return argv

    def launch_argv(self, slot: Slot) -> list[str]:
        """The full command, wrapper included."""
        wrapper = _expand(
            slot.host.launch_wrapper,
            device=slot.device, devices=slot.device_list, port=slot.port,
        )
        serve = self.serve_argv(slot)
        if not wrapper:
            return serve
        # A wrapper such as task-submit takes the command as one argument.
        return [*wrapper, " ".join(shlex.quote(part) for part in serve)]

    def stop_argv(self, slot: Slot) -> list[str]:
        if slot.host.stop_command:
            return _expand(
                slot.host.stop_command,
                device=slot.device, devices=slot.device_list, port=slot.port,
            )
        # Default: kill whatever is serving on that port. Matches on the port
        # rather than the model, so a relaunch with different flags is still hit.
        #
        # The character class is not decoration. `pkill -f` matches against every
        # command line including the one that invoked it -- over ssh that is the
        # remote shell carrying this very pattern -- so a literal pattern makes
        # pkill kill its own parent and the replica survives. Writing one
        # character as a class means the pattern no longer matches its own text,
        # while still matching the process we are after.
        return ["pkill", "-f", f"pypto_serving[.]cli .*--port {slot.port}( |$)"]

    def launch_env(self, slot: Slot) -> dict[str, str]:
        return {
            key: value.format(device=slot.device, devices=slot.device_list, port=slot.port)
            for key, value in slot.host.env.items()
        }

    async def launch(self, slot: Slot) -> LaunchedReplica:
        """Start a replica on an already-allocated slot."""
        transport = self._transport_factory(slot.host)
        try:
            await transport.run(
                slot.host, self.launch_argv(slot), self.launch_env(slot),
                detach=True, log_path=slot.log_path,
            )
        except LaunchError:
            self.release(slot.name)
            raise
        logger.info(
            "launched replica %s on %s device(s) %s (port %d); it will report ready when "
            "its model has loaded. Startup output: %s",
            slot.name, slot.host.name, slot.device_list, slot.port, slot.log_path,
        )
        return LaunchedReplica(slot=slot, spec=slot.replica_spec())

    async def signal_stop(self, slot: Slot) -> None:
        """Ask a replica to stop. Does not release the slot.

        Signalling is not stopping: a stop command returns as soon as the signal
        is delivered, while the replica takes seconds more to shut its worker
        down and let go of its port. The slot stays taken until the caller has
        confirmed it is really gone, or a relaunch could race the port.
        """
        transport = self._transport_factory(slot.host)
        try:
            await transport.run(slot.host, self.stop_argv(slot), {}, detach=False)
        except LaunchError as exc:
            # A stop command that fails leaves the device in an unknown state.
            # Reported, not raised: refusing to ever reuse the slot is worse.
            logger.warning("stopping replica %s failed: %s", slot.name, exc)


@dataclass
class FleetState:
    """What the router launched, so a restart can find it again."""

    replicas: list[dict] = field(default_factory=list)
