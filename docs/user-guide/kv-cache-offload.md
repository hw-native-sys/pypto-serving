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
