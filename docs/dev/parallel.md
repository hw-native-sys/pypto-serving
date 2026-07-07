# Parallel Serving Dev Notes

Serving supports a v1 `DP x TP` device topology. Data parallelism creates one
independent serving engine per replica, and tensor parallelism passes one device
group to the PyPTO L3 distributed worker for that replica. Single-device serving
remains the default. Pipeline parallelism and expert parallelism are not
supported yet.

## DP=2 Serving

For example, run two data-parallel replicas on devices 0 and 1:

```bash
python -m python.cli.main \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --devices 0,1 \
  --dp 2 \
  --tp 1 \
  --max-model-len 512 \
  --max-new-tokens 16 \
  --port 8899
```

Send a completion request to the DP=2 server:

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":16,"temperature":0.0}'
```

Offline `npu_generate.py` supports `--devices` and `--tp` for one logical TP
replica. It intentionally rejects `--dp > 1`; launch separate offline jobs if
data-parallel offline generation is needed.
