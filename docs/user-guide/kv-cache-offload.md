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

The example requests **200 decimal GB (186.26 GiB)**, not a model requirement or
a safe default for every host. The pool is allocated/prefaulted up front. Leave
separate host-memory headroom for weights and staging, runtime heaps, Python and
checkpoint metadata, concurrent processes and the operating system; do not set
the pool from `MemAvailable` alone or assume a container has a dedicated host.

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

## Sizing a retention target

For each **unique** page in the union of the desired recovery sets, charge
`TP * ceil(component_bytes_per_owner / system_page_size) * system_page_size`.
Sum every component in the group before rounding: `idx` includes KV and scale.
Within one partition, overlapping checkpoints and shared prefixes count their
common pages once; TP replicas and copies in different partitions count separately.
Full-history KV is a union, not one full-history copy per checkpoint. Rolling KV
and state require the union of each retained checkpoint's complete windows.

For example, use 43 layers (20 C128, 21 C4), 4 KiB host pages, P DP4/TP4, and
eight independent 65536-token prefixes in each of four partitions. Retain the
eight 8192-token periodic checkpoints of each prefix. Per prefix/per owner:

| Group | Unique pages | Useful bytes/page | Charged bytes/page |
| --- | ---: | ---: | ---: |
| `cmp_c128` | 512 | 20480 | 20480 |
| `cmp_c4` | 512 | 688128 | 688128 |
| `idx` (KV + scale) | 512 | 88704 | 90112 |
| `ori` (eight 128-token windows) | 32 | 1409024 | 1409024 |
| `hca_state` | 128 | 655360 | 655360 |
| `csa_state` | 32 | 344064 | 344064 |
| `csa_inner_state` | 32 | 86016 | 86016 |

This recovery-set union costs **551682048 bytes/owner (526.125 MiB)**,
**2.055 GiB with TP4**, or **65.766 GiB for 32 independent prefixes**.
It assumes these checkpoints were complete and eligible for publication.
Extra tail/output pages, additional prefixes and in-flight work need allowance;
capacity pressure may retain fewer checkpoints. A shared 32768-token prefix
reduces the charge only where its pages are actually shared in the same partition.
Decode-only PD keeps shareable history, not P's rolling/state checkpoint set:
the same 65536-token history is 390 MiB with TP1, before new Decode pages.
Request-local final state and bootstrap still need HBM and arrive from P.

The startup largest-page check is only a necessary allocation check. It does
not guarantee that a complete checkpoint fits or that a restore has enough
destination HBM pages, COW replacements and snapshot working space. Restore uses
the existing cache-manager allocator and can wait/fail under HBM pressure.
Reduce the retention target or leave more HBM headroom when needed; the configured
context/sequence ceilings do not reserve all possible full-length requests.

Tune using completed STORE/LOAD bytes, queue/source-pin age and bytes, actual
HBM-hit versus DDR-restored tokens, allocation waits/preemption and latency tails.
For host memory use instance/cgroup `memory.current`, `memory.peak`, `memory.stat`
and process PSS/Private, plus host `MemAvailable` and swap pressure. A shared HOST
mapping appears in every owner's RSS: summing those RSS values overcounts backing.
Budget for metadata outside the pool, including page keys, checkpoint tuples and
their memberships; short intervals and long contexts increase that overhead.

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

### Batch limits and native execution

The **256 MiB** payload cap is a software default per owner/batch, not a hardware
or SDK limit. It was selected from exploratory 1/4/16/64/256 MiB Ascend A3 DSpark
page-layout sweeps as a useful large batch that amortizes RPC/submission work.
It is the largest tested default in that sweep, not a global optimum. Owners
submit currently available work immediately; they do not wait to fill 256 MiB.
Validate again for different CANN versions, hardware, geometry and workloads.

The **32768 compact spans** cap independently bounds command metadata. On the
tested 64-bit platform its preallocated six-`size_t` table is 1.5 MiB per owner.
It counts compact component spans before layer expansion, not ACL commands or
pages; it is a storage/work bound rather than an empirically optimal span count.
Either cap can end a batch. Actual useful payload and native command counts
must be measured instead of assuming every batch is one contiguous DMA.

C++ is used for the small owner-local planning/submission loop, not to replace
Buffer ownership with a new runtime. In exploratory A3 same-layout strided
microbenchmarks, Python per-command 2D submission was about 1.1 GB/s, versus
about 18 GB/s for a C++ loop on registered THP backing; the batch API was about
2.8 GB/s on that layout. These are submission-path diagnostics, not model
throughput or guaranteed bandwidth. Keeping expansion and ACL calls in one
ctypes call also releases the GIL. This does not claim Python lacks async copy
APIs. Simpler continues to own allocation/context lifetime; Serving validates
model spans and copy proofs. Moving generic strided batch execution into a shared
runtime is separate follow-up work, not an SDK modification in this feature.

### Ownership boundaries and pressure

`KvCacheManager` owns HBM identities, allocation, snapshots/COW, prefix matching
and publication. `TieredCache` bridges those existing operations to DDR: it
reserves restore destinations, coalesces readers and returns cache-manager pins
only after confirmed completion. `CacheTier` owns DDR page keys, checkpoint
membership, storage references and eviction. The backend owns physical DDR
extents/transport jobs; the owner executes copies and never owns the tensor's
allocation. DDR readiness alone does not authorize physical HBM reuse.

The STORE queue has no independent byte or entry admission limit. Unique source
pages remain pinned in the existing finite HBM pools, so queue data cannot grow
past those pools, but pinning can exhaust their allocatable pages. The DDR budget
does not bound HBM pin pressure or guarantee progress/latency for new requests.
Background native completion and reader release allow pins/eviction to progress;
UNKNOWN preserves isolation instead of pretending that space was returned.
Report normal pending pins separately from quarantine. There is no optional-save
dropping or automatic admission policy in this revision.

The adapter currently uses cache-manager internals for grouped tables and
checkpoint publication. Follow-up interface work should expose those operations
through a small cache-manager API and derive inference/PD/DDR byte views from
one model storage description. Transport registration, final/bootstrap handoff
and DDR checkpoint eligibility should remain adapter responsibilities. Naming
and generic copy-runtime placement can be tightened in that work without
introducing a second allocator or resource state machine.

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
