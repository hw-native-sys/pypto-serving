# DeepSeek V4 MTP

PyPTO Serving supports DeepSeek V4 Flash through a converted W8A8 compressed-tensors checkpoint. The MTP deployment below uses eight devices with overlapped attention DP and MoE EP. [DSpark deployment](dspark.md) uses TP4 groups on eight or 16 devices.

## MTP Requirements

- A converted W8A8 compressed-tensors checkpoint.
- Exactly eight NPU device IDs.
- `--dp 8 --ep 8 --tp 1` for HTTP serving.
- `--block-size 128`.
- `--long-prefill-token-threshold 2048` or another explicit long-prompt dispatch limit up to the DeepSeek V4 8192-token main-prefill ceiling.

See [Checkpoint Conversion](#checkpoint-conversion) before starting a serving run with a source checkpoint that has not been converted to the PyPTO W8A8 layout.

## Checkpoint Conversion

PyPTO Serving expects a DeepSeek V4 W8A8 compressed-tensors checkpoint. The DeepSeek V4 Flash source checkpoint variant validated by this repository mixes FP8 weights with packed MXFP4 expert weights, so it must be converted before serving.

The conversion can run on CPU and does not require `torch_npu`. The source and output directories must be different, and the host must have enough free disk space for both copies.

Install the conversion dependencies using the PyTorch build validated for your Ascend environment:

```bash
python -m pip install --upgrade huggingface_hub safetensors
python -c "import torch, safetensors; print(torch.__version__)"
```

Download the source checkpoint, or use an existing local snapshot:

```bash
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /path/to/DeepSeek-V4-Flash
```

Preview the conversion, then run it:

```bash
python scripts/convert_deepseek_v4_to_w8a8.py \
  --input-dir /path/to/DeepSeek-V4-Flash \
  --output-dir /path/to/dsv4-flash-w8a8 \
  --dry-run

python scripts/convert_deepseek_v4_to_w8a8.py \
  --input-dir /path/to/DeepSeek-V4-Flash \
  --output-dir /path/to/dsv4-flash-w8a8
```

The converter writes one safetensors shard at a time using atomic replacement. If interrupted, rerun the conversion with `--resume` to validate and skip completed shards. `--input-dir` and `--output-dir` are required; `--dry-run` validates the source and prints the plan without writing files.

A successful run prints `Conversion complete` and leaves a converted `config.json`, `model.safetensors.index.json`, safetensors shards, and a `.pypto-w8a8-conversion.json` marker in the output directory.

After conversion, [Prepacked Weights](#prepacked-weights) describes the optional sidecar that reduces repeated startup work.

## 8-Device Offline Generation

The offline entry uses the same scheduler, worker process, rank-partitioned cache pools, and MTP acceptance path as HTTP serving, without opening a port.

```bash
PYPTO_RUNTIME_LOG=error \
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7 \
  --dp 8 \
  --ep 8 \
  --tp 1 \
  --block-size 128 \
  --max-model-len 512 \
  --long-prefill-token-threshold 2048 \
  --ring-dep-pool 131072 \
  --ring-task-window 131072 \
  --ring-heap 2147483648 \
  --generate-config '{"max_new_tokens": 20}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Repeat `--prompt` to exercise continuous batching, or add `--profile --profile-output /path/to/profile` to capture only the generation window after model initialization.

## 8-Device DP/EP Serving

Use the converted W8A8 checkpoint and run with overlapped attention DP=8 and MoE EP=8. Both parallel axes use the same eight physical ranks, so this is one model replica rather than eight independent serving replicas:

```bash
PYPTO_RUNTIME_LOG=error \
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --served-model-name dsv4-flash-w8a8 \
  --backend npu \
  --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7 \
  --dp 8 \
  --ep 8 \
  --tp 1 \
  --block-size 128 \
  --max-model-len 512 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 512 \
  --long-prefill-token-threshold 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --no-enable-prefix-caching \
  --ring-dep-pool 131072 \
  --ring-task-window 131072 \
  --ring-heap 2147483648 \
  --port 8225 \
  --show-startup-logs
```

Each NPU runs one prefill row at a time, so DP=8 admits up to eight prefill requests in one global step. The vLLM-style `--speculative-config` selects `method="mtp"`; `num_speculative_tokens` is the maximum number of draft tokens, and any positive value enables MTP. The 16-row MTP decode tile uses B8S2 for K=1, B4S4 for K=2-3, and B2S8 for K>=4. K values larger than seven are supported through repeated target verification chunks. Set `--max-num-seqs` no higher than 64, 32, or 16, respectively. Non-MTP decode retains B8S1T8. The deprecated `--num-speculative-tokens K` flag remains as a compatibility alias.

The server applies DeepSeek V4's model-specific message encoding for chat requests because the checkpoint does not ship a Jinja chat template. Chat mode is the default:

```bash
curl --noproxy "*" http://127.0.0.1:8225/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":32}'
```

Pass `"reasoning_effort":"high"` or `"chat_template_kwargs":{"enable_thinking":true}` to use the model's thinking prompt. An explicit `enable_thinking` value takes priority when both are given. The API supports text messages with system, user, developer, assistant, and `latest_reminder` roles, plus assistant function calls and tool-result messages. See [Tool Calling](../../user-guide/tool-calling.md) for the client workflow and [Function Tool Format](index.md#function-tool-format) for DeepSeek-specific limits. Multimodal content is not exposed by the request schema. Generation defaults to greedy sampling (`temperature=0`).

The main prefill kernel accepts a dynamic request extent up to 8192 tokens and walks it internally in 128-token tiles. The effective dispatch extent is the minimum of 8192, `--max-num-batched-tokens`, `--long-prefill-token-threshold`, and `--max-model-len`. An 8191-token prompt can therefore use one 8192-row main-prefill dispatch when those configured limits permit it, instead of 64 serving dispatches.

For repeated launches, set `PYPTO_CACHE=1` and point `PYPTO_CACHE_DIR` at a
shared artifact directory. PyPTO validates source, specialization and toolchain
identities before reusing generated code and binaries. Leave
`PYPTO_PROG_BUILD_DIR` unset for reuse: explicitly requesting build output
bypasses caching. The old named build slots are not reused. See the
[JIT cache guide](../../developer-guide/jit-cache.md) for read-only mode,
configuration precedence and cold-start costs.

For MTP K greater than 1, prefix caching is disabled automatically. For routine DeepSeek V4 serving, pass `--no-enable-prefix-caching` unless you are explicitly validating prefix-cache behavior.

## Prepacked Weights

DeepSeek V4 hidden-layer weights can be converted once into the rank-stacked host layout consumed by the serving runner. The optional sidecar reduces repeated startup work on later launches.

### Build the Sidecar

After converting the checkpoint, build the optional sidecar with the installed command:

```bash
pypto-prepack-deepseek-v4 /path/to/dsv4-flash-w8a8
```

The default sidecar path is `pypto-deepseek-v4-stacked-r8.safetensors` beside the checkpoint. `--output PATH` selects another path, and `--ranks N` sets the packed rank count (default `8`). Use `--force` to replace an existing sidecar after changing checkpoint shards or the packed rank layout.

### Runtime Behavior

On startup, a hot, valid sidecar is memory mapped instead of repacking hidden layers. A cold, missing, or stale sidecar falls back to the checkpoint path. Rebuild it with `--force` after replacing shards or changing the packed rank layout. The [sidecar contract](mtp-internals.md#prepacked-sidecar-contract) describes validation and layout details.

## Completion Check

Check server health first:

```bash
curl --noproxy "*" http://127.0.0.1:8225/health
```

Then send a deterministic completion request:

```bash
curl --noproxy "*" -s http://127.0.0.1:8225/v1/completions -H "Content-Type: application/json" -d '{"model":"dsv4-flash-w8a8","prompt":"Huawei is","max_tokens":25,"temperature":0.0}'
```
