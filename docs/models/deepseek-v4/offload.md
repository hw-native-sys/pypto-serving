# DeepSeek V4 expert offload

Expert offload runs the W8A8 V4 Flash checkpoint with routed experts in mapped
Host DDR and a fixed HBM expert cache per rank. Attention, shared experts, and
other model weights remain resident. Prefill and decode use one persistent
expert cache and the same model computation as the resident path.

## Run

```bash
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --platform a2a3 --devices 0,1 --dp 2 --ep 2 --tp 1 \
  --moe-offload-bytes 34359738368 \
  --max-num-seqs 8 --max-model-len 8192 --block-size 128 \
  --enable-chunked-prefill --long-prefill-token-threshold 1024 \
  --no-enable-prefix-caching \
  --prompt 'Briefly explain how processes differ from threads.' \
  --generate-config '{"max_new_tokens": 4096, "temperature": 0}'
```

`--moe-offload-bytes` is the HBM budget on **each rank**, in bytes. Zero selects
resident inference. A positive budget supports EP1, EP2, or EP4 on A2/A3, with
the same DP size and exactly that many device IDs. Speculative MTP and DSpark
are not supported with this option. The cache budget must hold all experts in
one prefill wave; startup rejects smaller budgets instead of shrinking the
wave or changing the requested budget. Each prefill dispatch accepts at most
`EP * 1024` tokens, tiled in groups of 128.

The host needs space for the full routed-expert backing as well as checkpoint
loading. The six packed INT8 weight and FP32 scale fields are copied once into
mapped Host DDR. The prepacked sidecar must match the selected rank count;
see [weight preparation](mtp.md#checkpoint-conversion).

## Dependency pairing

This path requires PyPTO's mapped expert dispatch and specialization support.
The compatibility candidate is PyPTO `f2b7261c3`, which retains the `fc73e265c`
compiler sources and pins runtime `80fb51f30`, with PTOAS `v0.78` and runtime-owned PTO-ISA
`45435c0624c1d0d056bf38aa60d4f56dc0ef3b6b`. The runtime extends `a5349f42c` with
immutable metadata image reuse; each invocation still resets mutable metadata
on the device. Build this runtime and use the version and checksum in
`toolchain/versions.env`; do not combine an unrelated installed runtime with
these compiler sources.

## Performance measurement

Keep model weights, prompt token IDs, expert budget, context length, precision,
warmup, and engine lifetime fixed. For comparison with the EP2 workload, use
32 GiB per rank, a C8 natural-EOS warmup, then C1 generation with an 8192-token
context limit. Exclude initialization and prefill from decode timing. Report
the full first-to-last-token interval, `(last_time - first_time) / (tokens - 1)`,
alongside the interval median, p95, maximum, token count, and termination cause.
Collect profiles separately from the reported timing run.
