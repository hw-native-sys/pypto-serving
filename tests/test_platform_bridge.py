"""Integration test for PlatformBridge (no NPU required).

Must be launched via mpirun so MPI_Init has a valid communicator:

    mpirun -np 1 --oversubscribe python3 -m pytest tests/test_platform_bridge.py -v

The test uses a dummy processFc that echoes the input back (no real
model), so it runs on any machine with the C++ platform built.
"""
from __future__ import annotations

import pickle
import time

from dataclasses import dataclass, field

import pytest


# ── module-level dataclasses (needed for pickle across threads) ───────────────

@dataclass
class _FakeRequest:
    request_id: str
    prompt_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_prompt_tokens(self):
        return len(self.prompt_token_ids)


@dataclass
class _FakeScheduledRequest:
    request: _FakeRequest
    num_new_tokens: int
    is_prefill: bool
    num_computed_tokens: int = 0
    block_ids: list[int] = field(default_factory=list)


@dataclass
class _FakeSchedulerOutput:
    scheduled_requests: list[_FakeScheduledRequest] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0


# ── helpers ──────────────────────────────────────────────────────────────────

# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def echo_bridge():
    """Session-scoped Bridge whose processFc echoes the payload back unchanged.

    Session-scoped because MPI can only be initialized once per process.
    """
    try:
        import serving_platform
    except ImportError:
        pytest.skip("serving_platform C++ extension not built — run `ninja -C platform/build python/serving_platform`")
    bridge = serving_platform.Bridge(lambda raw: raw, 4, 4096, 4096)
    yield bridge
    bridge.shutdown()


# ── tests ────────────────────────────────────────────────────────────────────

def test_single_roundtrip(echo_bridge):
    payload = pickle.dumps({"hello": "world", "n": 42})
    seq_id = echo_bridge.submit(payload)
    got_id, result = echo_bridge.getResult(timeoutMs=2000.0)
    assert got_id == seq_id
    assert pickle.loads(result) == {"hello": "world", "n": 42}


def test_sequence_ids_are_unique(echo_bridge):
    n = 10
    ids = [echo_bridge.submit(pickle.dumps(i)) for i in range(n)]
    assert len(set(ids)) == n, "sequence IDs must be unique"
    results = {}
    for _ in range(n):
        sid, raw = echo_bridge.getResult(timeoutMs=5000.0)
        results[sid] = pickle.loads(raw)
    assert len(results) == n


def test_throughput_overhead(echo_bridge):
    """Baseline vs platform latency — prints but does not assert ratio.

    The platform MPI loopback adds ~1 ms per job.  This test just
    documents the overhead at runtime so CI output captures it.
    """
    n = 50
    dummy = pickle.dumps(b"x" * 256)

    # Baseline: direct Python call
    def _direct(raw: bytes) -> bytes:
        return raw

    t0 = time.perf_counter()
    for _ in range(n):
        _direct(dummy)
    baseline_ms = (time.perf_counter() - t0) * 1000.0

    # Platform path
    t0 = time.perf_counter()
    for i in range(n):
        echo_bridge.submit(dummy)
    for _ in range(n):
        echo_bridge.getResult(timeoutMs=5000.0)
    platform_ms = (time.perf_counter() - t0) * 1000.0

    overhead_ms = (platform_ms - baseline_ms) / n
    print(
        f"\n[Baseline]  {n} calls  | {baseline_ms:.3f} ms total | {baseline_ms/n:.4f} ms/call"
        f"\n[Platform]  {n} jobs   | {platform_ms:.3f} ms total | {platform_ms/n:.4f} ms/call"
        f"\n[Overhead]  +{overhead_ms:.3f} ms/call"
    )


def test_pickling_scheduler_output(echo_bridge):
    """Round-trip a realistic SchedulerOutput-shaped object."""
    # Classes must be at module level for pickle to work across threads.
    out = _FakeSchedulerOutput(
        scheduled_requests=[
            _FakeScheduledRequest(
                request=_FakeRequest("req-1", list(range(128))),
                num_new_tokens=128,
                is_prefill=True,
                block_ids=list(range(8)),
            )
        ],
        num_prefill_tokens=128,
    )

    raw    = pickle.dumps(out)
    seq_id = echo_bridge.submit(raw)
    _, res = echo_bridge.getResult(timeoutMs=2000.0)

    recovered = pickle.loads(res)
    assert recovered.scheduled_requests[0].request.request_id == "req-1"
    assert recovered.num_prefill_tokens == 128
