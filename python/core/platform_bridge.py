"""Python wrapper around the C++ serving_platform.Bridge.

Replaces the mp.Queue + subprocess pattern of WorkerProcess with the
platform's coordinator→replica MPI channels. The processFc callback
receives a pickled SchedulerOutput and returns a pickled StepOutput.

The process must be launched via mpirun -np 1 (or mpirun -np N for
multi-rank — not yet wired here) so that MPI_Init has a valid
communicator to work with.
"""
from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scheduler import SchedulerOutput
    from .types import StepOutput


class PlatformBridge:
    """Coordinator+replica bridge backed by the C++ platform.

    Usage::

        bridge = PlatformBridge(worker)
        seq_id = bridge.submit(scheduler_output)
        seq_id, step_output = bridge.get_result(timeout_ms=5000.0)
        bridge.shutdown()

    The ``worker`` must already have called ``init_device_and_model()``
    before the bridge is created (model loading is the caller's
    responsibility).
    """

    def __init__(
        self,
        worker,
        *,
        payload_capacity: int = 4,
        payload_bytes: int = 262144,
        result_bytes: int = 65536,
    ) -> None:
        import serving_platform  # noqa: PLC0415

        self._worker = worker

        def _processFc(raw: bytes) -> bytes:
            scheduler_output = pickle.loads(raw)
            step_output = worker._execute_step(scheduler_output)
            return pickle.dumps(step_output)

        self._bridge = serving_platform.Bridge(
            _processFc,
            payload_capacity,
            payload_bytes,
            result_bytes,
        )

    def submit(self, scheduler_output: SchedulerOutput) -> int:
        """Pickle and submit one scheduler batch. Returns the sequence ID."""
        raw = pickle.dumps(scheduler_output)
        return self._bridge.submit(raw)

    def get_result(self, timeout_ms: float = 5000.0) -> tuple[int, StepOutput]:
        """Wait for one result. Returns (seq_id, StepOutput).

        Raises RuntimeError on timeout.
        """
        seq_id, raw = self._bridge.getResult(timeout_ms)
        return seq_id, pickle.loads(raw)

    def shutdown(self) -> None:
        self._bridge.shutdown()
