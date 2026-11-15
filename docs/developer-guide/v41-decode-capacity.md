# V4.1 decode capacity

Set `PYPTO_V41_DECODE_BATCH_PER_DP` before starting the serving process to
specialize V4.1 decode for a smaller request capacity. The accepted range is
1 through 12. The default is 12 requests per DP partition.

For serial K5 latency measurements, use

```bash
PYPTO_V41_DECODE_BATCH_PER_DP=1 pypto-serving \
  --max-num-seqs 1 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  <model and device arguments>
```

Each request requires six target rows. Capacity one uses a 16-row padded decode
extent, while capacity twelve uses 80 rows. Serving configures the lib dimensions
before importing model entries and allocates matching host and device buffers.
The selected capacity also bounds request state slots and draft cache storage.
Smaller capacities require the configured per-partition request capacity and
cache request slots to fit that capacity. DP4 with capacity one can hold up to
four simultaneous requests when each partition has one cache request slot.

The shared MoE extent also controls prefill's internal MoE chunking. Measure
TTFT together with TPOT when choosing a capacity. Keep the prefill token budget
explicit in performance comparisons.

Start a fresh worker to change capacity. Compilation and JIT cache lookup use
the resulting entry signatures. Source archives and prebuilt programs must
agree with the selected tensor shapes before execution.
