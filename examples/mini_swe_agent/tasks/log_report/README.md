# Serving log report repair exercise

Fix this small command-line tool without changing its public interface or tests.
Only edit `report.py` and `cli.py`. It uses the Python standard library.

Each nonblank line in an input JSONL file is a request with `model`, `status`,
and numeric `latency_ms`. Inputs are well-formed; status is `ok` or `error`.

Required behavior:
- `python cli.py access.jsonl` reports all requests.
- `--model NAME` selects only exact matches; without it, include all models.
- `requests`, `successful`, and `failed` count the selected requests.
- `mean_ms` and `p95_ms` use only successful requests.
- P95 uses the nearest-rank definition: sort N latencies and use rank
  `ceil(0.95 * N)`, with ranks starting at 1 (zero-based index
  `ceil(0.95 * N) - 1`). Do not interpolate.
- If there are no successful requests, both latency statistics are zero.
- Empty input and filters with no matches produce zero counts and statistics.
- Ignore blank input lines; keep fractional latency values.
- Print one JSON object to stdout with the existing five keys.

First reproduce the failure with `python -m unittest -v`, inspect the source,
fix the underlying issues, rerun tests, and explain the changes and validation.
