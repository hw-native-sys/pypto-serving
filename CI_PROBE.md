Throwaway CI probe (do not merge).

Purpose: reproduce a local Qwen3-14B warmup fault under CI conditions.
Local stack (pypto main 04f1f510 built locally, CANN 9.0.0) fails during
"[warmup] prefill dispatch" with orch_error_code=7 REQUIRE_SYNC_START_INVALID
(require_sync_start block_num exceeds the physical core count), deterministically
across devices, with compile config identical to serving main. The serving CI
builds pypto from default-branch HEAD with the runner toolchain bundle; this PR
checks whether the same fault appears there.

Close after reading the "Run Qwen3 accuracy guard" result.
