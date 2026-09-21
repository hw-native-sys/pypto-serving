# Phase H PC1 dual-node acceptance

This directory contains the **prepared but intentionally unexecuted** H7
acceptance gate. H0–H6 are Host/code-ready; do not call PC1 end-to-end complete
until this gate runs on two fully available 16-device nodes.

## Runtime profile

Use the normal shared PD JSON and add exactly one runtime field:

```json
{
  "runtime": {
    "prefix_cache_mode": "d_only",
    "prefill": [{"host": "PREFILL_HOST", "port": 8101}],
    "decode": [{"host": "DECODE_HOST", "port": 8102}]
  }
}
```

The shared profile is authoritative. The P engine derives Prefix Cache
disabled and the D engine derives Prefix Cache enabled; no extra launcher flag
is required. `provider` remains `mooncake` unless explicitly configured.

## H7 sequence

1. Capture P and D `/internal/pd/metrics` as `before.json`.
2. Send one K7 request with an identical prompt of at least 128 tokens and
   capture metrics as `after-cold.json`.
3. Send the same request again and capture metrics as `after-hit.json`.
4. Compare output token IDs/digest with the non-PD K7 reference.
5. Run `verify_pc1_metrics.py` on the three composite snapshots.
6. Repeat for partial hit, full hit, final-state-only, cancellation and UNKNOWN.

Each composite snapshot has this address-free shape:

```json
{
  "prefill": {"counters": {"transfer.bytes": 0}},
  "decode": {
    "counters": {
      "prefix.requests": 0,
      "prefix.cold_requests": 0,
      "prefix.hit_requests": 0,
      "prefix.hit_tokens": 0
    }
  }
}
```

The cold request must add one cold reservation. The repeat request must add one
hit reservation, add at least 128 hit tokens, and transfer fewer bytes than the
cold request. A full KV hit must still transfer nonzero final-state bytes and
reach normal commit/Decode completion.

Do not use mock results as H7 evidence. Do not reset devices or stop unrelated
processes to obtain the test window.
