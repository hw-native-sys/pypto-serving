# DeepSeek V4

PyPTO Serving runs DeepSeek V4 Flash from a converted W8A8 checkpoint. Choose the execution path that matches your device topology and decoding workload:

| Path | Devices | Parallelism | Decode |
| --- | --- | --- | --- |
| [MTP](mtp.md) | 8 | `--dp 8 --ep 8 --tp 1` | Autoregressive or MTP speculative decoding. |
| [DSpark](dspark.md) | 8 or 16 | TP4, with `--dp 2 --ep 8` or `--dp 4 --ep 16` | Target-only K=0 or DSpark speculative K=7. |

Both paths use the [checkpoint conversion](mtp.md#checkpoint-conversion). The optional prepacked sidecar described in the MTP guide is specific to that path; DSpark loads its own weight layout. For the client workflow, see [Tool Calling](../../user-guide/tool-calling.md); for structural constraints on DSpark K=7, see [Tool Constraints](tool-constraints.md).

## Function Tool Format

DeepSeek V4 encodes tool definitions and history in its DSML message format. It accepts text messages with system, user, developer, assistant, and `latest_reminder` roles. Tool results are encoded in the original assistant call order even if the client supplies results out of order. The parser preserves DSML parameter types without schema-based coercion or guessed JSON repairs.

Tool-history argument values cannot contain the reserved `</｜DSML｜parameter>` delimiter, including inside nested JSON values. Tool-result content cannot contain `</tool_result>`. These inputs return HTTP 400 before generation because they would break the model's history encoding. In unconstrained `auto`, a model-generated function name can be absent from the request's declared tools; clients must validate names and arguments before execution.

## Token Limit Behavior

`min_tokens` is honored in autoregressive decoding without speculative MTP and in DSpark. Speculative MTP accepts the field but currently uses an effective minimum of zero. With DSpark structural constraints, EOS can be emitted before `min_tokens` when the grammar allows no other token, so a constrained request cannot stall on an empty token set.

Implementation details are in [MTP Internals](mtp-internals.md) and [DSpark Internals](dspark-internals.md).
