# DeepSeek V4.1 A5 decode scheduling investigation

Measurements on 2026-11-16 compare the complete K5 decode paths on four Ascend 950DT devices. The investigation has established common timing groups and rejected three configuration candidates. A reduction of the task orchestration gap remains unverified.

The measured PyPTO stack uses serving `63654c91d36aa0d8d6c6d846da55f11c76584a14`, lib `a658d93f53bdc54b13810c1ac70de4d288614544`, PyPTO `950a30ec`, Simpler `31f9e72`, PTOAS 0.70 and the mmap, active-row and resident-scratch methods from serving #332. These are frozen benchmark inputs. This document is based on serving main `ec8b0636`, whose lib pin is `b67a5fd6`. Performance on that current dependency combination has not been measured here. In particular, the newer lib uses `hc_head_norm_store` in its MoE path.

DeepInfer uses source base `a679bb86ce36e883a98a97733492b686e5b9a08e` with local A5, CANN, graph and CPU-offload adaptations. Its working-diff SHA-256 is `895f380699e21e62043adaba7d922ef2602bbd458bfc6c4b40a170d3af212d04`. Source-file, auxiliary-loader and loaded-extension hashes are retained with the measurement artifacts.

| Setting | PyPTO | Adapted DeepInfer |
| --- | --- | --- |
| Model | Real DeepSeek V4.1-Flash checkpoint | Same checkpoint |
| Physical devices | 0, 1, 2, 3 | 0, 1, 2, 3 |
| Decode topology | TP1 / DP4 / EP4 | Attention TP1 equivalent / DP4 / EP4 / CP1 |
| Request concurrency | 1, serial | 1, serial |
| Request placement | Least-loaded round-robin cache partitions | Requests submitted to DP engine 0, other ranks use placeholders |
| Speculation | Fixed K5 | Fixed K5, confidence varlen disabled |
| Target tensor row extent | Capacity for 12 requests per DP, 72 target rows padded to 80 | Six rows on the active rank, one row for a placeholder graph |
| Target and draft execution | Persistent native ring runtime | Both graph replays enabled |
| CPU Engram | mmap batch gather and active rows | Graph-aware CPU offload |
| Prefill chunk budget | 80 | 512 |
| Python / Torch | 3.10.19 / 2.10 CPU | 3.12.15 / 2.13 CPU, torch_npu 2.13 rc1 |
| CANN | 9.3 | 9.3 |
| Torch CPU threads | 2 | 1 |

Both paths use the fixed GSM8K first-sixteen five-shot fixture, greedy sampling, seed 123, context limit 8320, KV block size 128, disabled prefix caching and output limit 256. Stops are `Question`, `Assistant:` and `<|separator|>`. Tokenized prompts contain 603–684 tokens.

The first three cases produce exactly matching input IDs, output IDs and stops across frameworks, with output lengths 57, 49 and 131. The full sixteen-case output sequences differ across frameworks. The common device profile therefore uses these three cases. Each request is fully prewarmed. Full six-row K5 rounds 5, 6 and 7 supply nine samples per rank. Corresponding masked ranks are selected using the active DeepInfer rank's full steps. Accepted draft sequences and intermediate context lengths can differ across frameworks.

The rank 0 slices include the request-placement and masking difference above. Across all four ranks, mean device envelopes span 87.336–91.509 ms in PyPTO and 15.356–15.364 ms in DeepInfer. Uncovered intervals span 26.201–31.150 and 1.002–1.779 ms respectively. Target MoE spans 29.476–29.964 and 5.369–6.698 ms.

The fixed PyPTO tensor extent is an additional workload difference. `MAX_BATCH_PER_DP=12` and six target rows per request produce an 80-row padded trunk even with request concurrency one. Active counts still govern parts of the computation. The effect of specializing that extent to the configured active batch needs its own numerical and performance comparison. The measured gap cannot be assigned solely to runtime scheduling. [Frozen lib capacity and entry](https://github.com/hw-native-sys/pypto-lib/blob/a658d93f53bdc54b13810c1ac70de4d288614544/models/deepseek_v4_1_flash/decode_fwd_dspark.py)

| Group | Common semantic boundary | PyPTO rank 0 mean, ms/round | DeepInfer rank 0 mean, ms/round |
| --- | --- | --- | --- |
| S01 | Host metadata and CPU Engram preparation, including observed waits | 8.675 | 7.338 |
| S02 | Target embedding, device Engram and input/cache metadata | 2.777 | 0.127 |
| S03 | Target mHC, per-layer normalization and MoE input quantization | 7.043 | 2.262 |
| S04 | Target attention, projection, compression, indexing and output projection | 14.761 | 3.847 |
| S05 | Target MoE, routing, transport, shared/routed FFNs and combination | 29.476 | 5.369 |
| S06 | Target final norm, heads, verification and accepted state | 5.509 | 0.482 |
| S07 | Target feature extraction and draft context/KV preparation | 1.820 | 0.074 |
| S08 | Draft mHC, per-layer normalization and MoE input quantization | 0.390 | 0.171 |
| S09 | Draft attention and MoE, including transport | 4.101 | 0.519 |
| S10 | Draft final norm, head, Markov sampling and state publication | 6.521 | 0.690 |

S01 measures host API wall time. S02–S10 measure unions of device task envelopes. Overlap exists within and across groups, and CPU work can overlap device execution. Adding the ten rows does not produce TPOT. Engram host APIs include synchronization time, which limits conclusions about pure CPU lookup speed.

DeepInfer directly emits FP8 MoE activations and scales from its mHC/norm path. PyPTO's separate `_gate_quant_vf` is included in S03/S08 to match that boundary. Attention input casting stays in attention on both paths. Inseparable fused operations stay with the corresponding group as a whole.

| Device timing boundary | PyPTO rank 0, ms/round | DeepInfer rank 0, ms/round |
| --- | --- | --- |
| First kernel start to last kernel completion | 91.509 | 15.364 |
| Union of task envelopes | 60.359 | 13.584 |
| Intervals without a recorded kernel task | 31.150 | 1.779 |

The uncovered intervals can include orchestration, synchronization and other waits. Their full duration cannot be assigned to AICPU execution from these records alone. Communication and waits inside a kernel are included in that kernel's envelope.

The PyPTO profile contains 2,126 clocked task IDs per sampled round. Its target MoE group contributes 761 graph tasks, including signal clearing, after input quantization is assigned to S03. DeepInfer records 1,370 hardware kernels per round, with 120 in target MoE. The respective units are native task submissions and CANN kernel records. The MoE task granularity difference is substantial and accompanies the 29.476 versus 5.369 ms group measurements.

PyPTO task IDs are joined to a separately recorded dependency graph of the same callable digest. The closest preceding graph capture is used for each timing dispatch. Draft-geometry dependency descendants distinguish reused target/draft functions. Generated orchestration confirms that the anonymous function ID 69 is the C1A compressor projection. Task envelopes span all participating cores of each task, matching the aggregate hardware-kernel boundary used for DeepInfer.

DeepInfer timing Events recorded inside graphs retained capture-time timestamps in a replay probe. The measurements instead use marker-free CANN task timestamps. A separate labeled capture supplies nested host MSTX ranges and capture-time HostTask/TaskInfo records. Labels transfer only after the complete kernel name/type sequence matches each raw graph. The auxiliary HCCL AICPU cache-eviction task is excluded from compute templates. The rank 0 eager metadata remainder is explicitly unmapped and contributes about 0.047 ms/round.

The raw PyPTO profiler copies and serializes records after device execution. Its instrumented host dispatch wall time includes that collection overhead. The uninstrumented streaming requests supply TTFT and TPOT. TTFT runs from request start to the first nonempty token event, including synchronous bootstrap. TPOT is the interval between first and last token events divided by emitted tokens minus one. Startup, compilation and weight upload are excluded.

| Common three-case streaming measurement | PyPTO, pipeline off | Adapted DeepInfer |
| --- | --- | --- |
| Formal requests | 12 | 3 |
| TTFT median | 1086.33 ms | 356.05 ms |
| TPOT median | 34.37 ms/token | 5.38 ms/token |
| Accepted/proposed draft ratio | 47.04% | 49.71% |
| Raw tokens per recorded verification/decode round | 3.352 | 3.485 |
| Delivered tokens after the first event per recorded round | 3.296 | 3.441 |

The repeated cases have exact matching input/output IDs and stops. DeepInfer's verification counter and its graph replay counter have different boundaries. Placeholder and pipeline-drain work remains represented by streaming wall time. The per-round ratios describe token production and do not replace event-based TPOT.

| Candidate | Device evidence | Decision |
| --- | --- | --- |
| AICPU threads 4 to 5 | 64 formal requests in 4/5/5/4 blocks. Paired TPOT change +1.816 ms/token, 95% case-cluster interval +1.343 to +2.870 | Withdrawn |
| AICPU threads 4 to 3 | 64 formal requests in 4/3/3/4 blocks. Paired TPOT change +5.459 ms/token, interval +3.792 to +6.214 | Withdrawn |
| Explicit software pipelining | 128 formal requests in four on/off/off/on processes. Paired TPOT change +0.159 ms/token, interval −0.390 to +0.638 | Withdrawn |
| Existing HBG backend | Actual prefill and complete K5 decode codegen reject mixed Host/Device control tensors | Blocked at the composite/runtime contract |

The software-pipeline comparison prewarms all sixteen cases and runs two formal passes per process. Each setting has 64 formal requests, 6,956 output tokens, 2,128 decode rounds, 4,844 accepted drafts and 10,640 proposed drafts. Every paired input/output sequence, stop reason and accepted/proposed draft sequence matches. Confidence intervals resample the sixteen cases with all four repeats per case, 10,000 times with seed 123.

| Uninstrumented metric | Pipeline off | Pipeline on |
| --- | --- | --- |
| TTFT median | 1109.36 ms | 1124.97 ms |
| TPOT median | 36.32 ms/token | 36.18 ms/token |
| Median paired TPOT change, on minus off | — | +0.159 ms/token |
| Median paired TPOT reduction | — | −0.415%, interval −1.792% to +1.035% |

Pooling medians and pairing the same cases answer different statistical questions. The paired result does not establish a benefit. The first two processes alone suggested a 1.41% paired reduction, which disappeared after the reverse-order controls. The option changes three generated decode `hc_collapse` C++ kernels and one drafter kernel. Every generated orchestration C++ file is byte-identical across the settings. The sampled uncovered interval is 31.150 ms with the option off and 32.152 ms with it on. The production candidate was removed after this check. Its existing CPU tests had passed all 131 cases.

The first HBG attempt changed RunConfig metadata while the frozen compiler still emitted ring-runtime code. Inspection of generated `RUNTIME_CONFIG` caught this mismatch. Correct runtime selection through `PassContext` then produced the actual prefill failure. A separate direct K5 compile independently produced the same error for `l2_decode_fwd_dspark`.

| HBG contract item | Evidence and required work |
| --- | --- |
| Required boundary | Complete CED prefill and recurrent K5 decode using graph-compatible control inputs |
| Existing capability | Native complete V4.1 lib entries compile and run on the ring runtime |
| Current failure | `host_build_graph requires separate tensor arguments for host tensor.read/write and device kernels` |
| External control data | Provide separate Host control values and Device data with explicit binding semantics |
| Device-produced control data | Define how route/receive counts drive later graph work, using supported device predicates, bounded tasks or staged synchronization |
| Validation level | CPU codegen failure for actual HBG. No HBG complete-model device acceptance |
| Impact | Graph-based reduction of the current scheduling gap remains blocked |

The current upstream `_host_tensor_args` implementation retains the mixed-use guard. Existing PyPTO #3195 discussion records the separate Host/Device input direction and the additional device-produced-control problem. Reusable graph regions also need an explicit graph-boundary contract beyond backend selection. [Current guard](https://github.com/hw-native-sys/pypto/blob/d0127ff79cc0e829e420cd01ffded5f4084cdadb/python/pypto/backend/_host_tensor_args.py) · [Existing control-data discussion](https://github.com/hw-native-sys/pypto/issues/3195#issuecomment-6572215112) · [Graph execution design](https://github.com/hw-native-sys/pypto/issues/2399)

Principal completed tasks are `task_20261116_013445_337157018987` for the common PyPTO profile and 4/3 control, `task_20261116_014832_38888462011` for same-card DeepInfer, `task_20261116_021003_80662332487` and `task_20261116_021941_119425822831` for the first pipeline pair, and `task_20261116_023737_186768511547` for the reverse-order pair. All completed with exit 0 and released their four devices. Full raw traces, bounded source/launch manifests, answers, counters and analysis scripts remain in the owner's dated measurement archive.
