# DDR prefix-cache offload

DSpark can retain immutable prefix-cache pages and their matching compression
state in host DDR. Ordinary serving, Prefill and Decode use the same cache
manager and offload implementation. The default is disabled. Contents are volatile
and are not recovered after a process restart.

Add these options when starting each model process:

```bash
--enable-prefix-caching --kv-offload-bytes 200000000000
```

For PD, configure each P/D process independently; the Router has no DDR pool.
`--kv-offload-bytes` is the physical budget for the entire instance, including
alignment and TP replicas, rounded down to a system page. It is not a per-rank
budget. The current backend is `--kv-offload-backend simpler`.

The backend requires Linux, the matching PyPTO/Simpler owner-service and retained
tensor interfaces, CANN's Ascend asynchronous copy APIs and `aclrtHostRegisterV2`, and enough host memory
for the configured pool. On a host with multiple NUMA nodes, `libnuma.so.1` and permission
to apply `mbind` to the pool mapping are required. The pool is interleaved across
allowed memory nodes before first touch; other allocations keep their existing
memory policy. Each persistent mapping requests transparent hugepages before
prefaulting and PINNED registration. Actual hugepage coverage depends on the host
configuration. Registration and physical allocation happen during startup.
A C++17 compiler (`CXX`, default `g++`) and CANN headers under `ASCEND_HOME_PATH`
are required when offload is enabled. A small temporary copy helper is built
before owners start and removed after they retire; disabled offload does not build it.

## Execution and ownership

One shared HOST backing is allocated per instance. Each rank owner imports and
registers it once. DDR extents for different ranks and requests never overlap
while live. Owner-local copy threads use a stream separate from computation;
neither the API thread nor the compute submission thread waits for native DMA.

New immutable full-history pages enter a metadata queue with their source HBM
references held. Rolling KV and compression state enter only as members of a
confirmed joint checkpoint; overlapping checkpoint windows share their pages.
Decode-only PD retains its shareable history, while active final state stays request-local.
An idle owner immediately batches available pages, up to 256 MiB of useful payload
and 32768 compact component spans. Its command table is allocated once at startup;
layer rows are expanded and planned inside the native helper. There is one batch in flight per owner. A batch has one final native
completion; a logical page becomes ready only after all required components and
TP copies complete. An index page includes both index KV and scale data.
The native planner compares layer rows within a page with regular runs across pages,
using the plan with fewer two-dimensional copies. Contiguous ranges use asynchronous memcpy. This changes command
grouping without changing page layout or copying gaps between authorized pages.
There is no HBM staging buffer, data packing, or selectable legacy copy path.

Completed batches notify the existing engine event loop to publish pages and
return copy references without waiting for a model step to return. Restore
waiters use completion notifications instead of periodic page scans. LOAD and
STORE still share one copy stream per owner.

Lookup uses the normal grouped prefix boundaries. A hit must include the required
KV and compression state. A missing HBM page can be restored from DDR into pages
reserved by the existing cache manager. Concurrent readers share restoration;
cancelling a reader cannot release another reader's resources. In PD, DDR restore
finishes before the affected destination pages become writable by the peer.
D matches only its shareable KV groups; request-local final state and bootstrap
data still arrive from P, including when D has a full KV hit.

DDR uses LRU eviction of eligible content. Readers and in-flight operations keep
their ranges until confirmed complete. There is no offload queue-length limit:
under pressure, queued stores can retain HBM pages and delay allocation for new
work. A finite DDR budget does not guarantee retention of every prefix.

Shutdown drains native work, unregisters the pool inside each owner, and then
closes the backing. An unknown completion keeps the corresponding ranges and
pins isolated; a timeout is not evidence that memory can be reused.

## Measurement

`kv_offload_ready` reports the effective pool budget; `kv_offload_restore` records
actual restores, and `kv_offload_closed` reports completed byte/page counters.
Debug logging adds per-batch completion summaries. Native bytes count TP replicas;
they must be distinguished from logical pages and token hit rates.
Native timing includes command planning, submission and stream completion; it is
not isolated DMA time.

Measure cold, HBM-hit and DDR-hit requests separately. Exclude startup, data
readback and correctness hashing from bandwidth measurements. Report Host step
time separately from kernel time, and use native completion rather than HTTP
success as the transfer boundary.

For lightweight, continuous diagnostics, set this before starting each instance:

```bash
export PYPTO_SERVING_KV_OFFLOAD_TRACE_DIR=/path/to/run/p/offload
export PYPTO_SERVING_DECODE_METRICS_DIR=/path/to/run/p/decode
```

Use separate directories for P/D or ordinary replicas. Offload tracing is disabled
when its variable is unset and creates no recorder when offload is disabled.
It writes one `offload.<pid>.<session>.jsonl` file for the instance. Request threads
enqueue metadata; a background writer performs serialization and file I/O.
The diagnostic queue is bounded: overflow is reported in `state`/`end` records,
and an incomplete capture cannot establish total bandwidth or complete coverage.
This bound does not limit the offload work queue. Recording failures do not change
native proof, cancellation, pin retirement or quarantine decisions.

Records include request restore start/ready/cancellation/failure; per-owner batch
identity, direction, partition, cache groups, physical bytes and initiating LOAD
request IDs; descriptor preparation, native planning/submission/completion and
RPC timestamps; and one-second residency/queue/cumulative-counter snapshots.
Batch timing separates endpoint lock acquisition, send/receive, owner receipt,
native execution and reply preparation. These are Host timestamps, not a measure
of Python scheduling or device utilization by themselves.
Shared restores list their original request IDs rather than counting a shared
transfer twice. STORE batches are shared cache work and have no invented owning
HTTP request. No prompts, cache contents, hashes or raw addresses are recorded.
Queue/source-pin lifetime includes metadata admission and completion consumption;
`store_inflight` and `unconfirmed_pages` are pending Host confirmation, not a
claim that the DMA engine is still active.
`store_reaped` and `load_reaped` mark Host consumption of confirmed transfers;
page publication, complete restore readiness and scheduling are separate events.

`restore_ready` means the candidate recovery set is readable, not that inference
succeeded or its resources retired. A cancelled reader can still leave native
LOADs in flight. `end.complete` describes the recording, not resource safety;
`shutdown.resources_closed` is emitted only after normal backend/pool retirement.

Enable recording once, then run a controlled probe and the full workload in the
same service. Give them separate client artifacts and prefix namespaces, and
record each phase's Unix start/end after its requests and offload jobs drain.
Compare counter deltas and completed batches within that window; do not reset
service counters or clear caches between phases. The file can be read live,
with buffered data flushed after an idle second. Phase boundaries must not cut
through in-flight batches. Check queue/restore/quarantine snapshots as well as
HTTP completion before starting the next workload.

Count TP replicas in physical bytes and report per-owner bandwidth alongside
instance totals. Use the union of overlapping native intervals on one host, not
their sum, for aggregate active-window bandwidth. Native submission can overlap
DMA, so the final completion wait is not the entire copy time. Monotonic clocks
are local to a host. Decode dispatch-to-reclaim spans can show Host-step envelope
overlap, but do not establish overlap with actual device kernels.

## Optional SSD cache

An optional local SSD tier retains the same reusable contents as DDR. Enable it
per model process with an existing directory:

```bash
--enable-prefix-caching --kv-offload-bytes 20000000000 \
--kv-ssd-bytes 30000000000 --kv-ssd-path /data/cache
```

SSD defaults to zero capacity and creates no SSD threads, files or work windows
when disabled. A positive capacity requires DDR offload and prefix caching.
Both budgets include alignment and TP replicas across the entire instance.
Two LOAD windows per rank are reserved inside the DDR budget, targeting 32 MiB
each. Smaller budgets reduce the window size while still fitting the largest
page. At 16 ranks, full-size windows consume 1 GiB of the stated DDR budget;
STORE uses the existing cached DDR pages rather than these windows.

The same hierarchy coordinator handles DDR and SSD lookup, source readers,
shared restores and cancellation. One logical checkpoint member index spans
the stores; each store retains its own physical allocations and completion proof.
Evicting one replica does not invalidate healthy copies in another tier.
Only contents eligible for DDR enter SSD. Reads reuse HBM and DDR first, reading
only missing members from SSD. File I/O runs in background threads; owner-local
H2D uses the existing compact copy queue. DDR promotion runs separately and does
not delay an otherwise completed HBM restore. Read references protect file slots
and DDR ranges until their users finish. LRU reclaims eligible SSD contents.
Re-publishing immutable contents already confirmed in SSD does not recapture
them from HBM merely to populate DDR. Missing or failed SSD copies remain
eligible for capture.

The directory must support direct I/O, preallocation and hole punching. Each
process owns a private file namespace, removed after its I/O drains. Contents are
volatile; there is no index recovery after restart.

SSD uses the same optional offload trace. `ssd_file` records one logical page's
file admission, executor dispatch, I/O start/end, useful/physical bytes and syscall
count. A vector call covers the TP views up to the system iovec limit; aligned
partial results advance both the file offset and remaining views. Admission to
dispatch includes capacity/extent preparation, while dispatch to I/O start is
executor waiting. The I/O span includes the Python loop and syscall waiting.

`ssd_handoff` records window assignment, staging start, all file reads settling
and confirmed native readiness. Its cache handles join `batch.work_handles`;
request IDs remain the original restore IDs. `ssd_promotion` reports independent
Host copy queue/execution, and `ssd_store_reaped` reports DDR reader retirement.
`state.tiers.ssd` adds bounded queue, window and physical-usage summaries without
scanning every file page. These intervals can overlap and must not simply be
summed as request latency. Errors or wakeups alone are not completion proof.

Normal shutdown drains I/O and readers, closes the data file descriptors and
removes the instance's private directory, unregisters owner mappings and releases
the shared DDR backing. SSD contents are not retained between runs.
`ssd_closed.files_removed` confirms file cleanup; `shutdown.resources_closed`
confirms normal backend/pool retirement. Verify both and process/file references
when qualifying cleanup. Deleted files held open by another process still occupy
space; HTTP shutdown or a zero cache counter alone does not prove release.
