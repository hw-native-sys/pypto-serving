# DeepSeek V4.1 image requests

The V4.1 adapter accepts OpenAI chat content containing text and `image_url`
blocks. Images may use HTTP(S) URLs or base64 data URLs. Image blocks belong
to user messages. Install the optional image dependency in the serving
environment with `pip install -e '.[vision]'`.

Start the A5 server using the complete DeepSeek-V4.1-Flash checkpoint, including
its vision, aligner, image delimiter and visual routing weights. A four-device
example is shown below. Set `MODEL_DIR` to the checkpoint directory and select
the allocated devices.

```bash
pypto-serving --model "$MODEL_DIR" --backend npu --platform a5 \
  --devices 0,1,2,3 --tp 1 --dp 4 --ep 4 \
  --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 80 \
  --ring-heap 2147483648 \
  --host 127.0.0.1 --port 8000
```

For K5, add `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`.
The 80-token example reserves 2 GiB per language-model runtime ring to leave
space for the separate vision workspace alongside resident DSpark resources.
Larger prefill chunks require their own workspace sizing and validation.
The model context limit includes expanded image positions and generated tokens.
Images can cross prefill chunk boundaries. Image encoding sizes each runtime
ring from the image grid and the lib encoder's temporary tensor shapes. The
outer scope retains feature and aligner buffers, and the layer scope retains
layer intermediates and tiled attention scratch. The other two rings reserve
16 MiB each. Reservations are logged per image and rounded to powers of two.
Set `PYPTO_V41_VISION_RING_HEAP` to a power-of-two byte count to override every
ring with the same reservation.

Send an image and a question in the same user message.

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://example.org/image.jpg"}},
      {"type": "text", "text": "What vegetable is shown in the image? Answer with one word."}
    ]
  }],
  "temperature": 0,
  "max_tokens": 32
}
```

Image encoding runs on the first prefill of each request. Later chunks reuse
the request's visual embeddings. Request release also releases these embeddings.
Prefix-cache keys include the image identity and its prompt position.
