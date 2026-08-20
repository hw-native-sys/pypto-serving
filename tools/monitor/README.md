# PyPTO Monitor

PyPTO Monitor is a local companion dashboard for PyPTO Serving. It polls the
structured metrics endpoint, keeps detailed recent samples and daily totals in
SQLite, and serves a browser dashboard without Prometheus, Grafana, or external
web assets.

Start PyPTO Serving normally, then launch the monitor from the repository root:

```bash
python -m tools.monitor \
  --target http://127.0.0.1:8899 \
  --port 9090
```

Open <http://127.0.0.1:9090>. The dashboard listener defaults to
`127.0.0.1`, so it is not reachable from other hosts unless `--host` is changed.

The default database is
`~/.local/state/pypto-serving/monitor.sqlite3`. Override it when running in a
container or when history needs to live on a persistent volume:

```bash
python -m tools.monitor \
  --target http://127.0.0.1:8899 \
  --database /var/lib/pypto-monitor/metrics.sqlite3 \
  --timezone Asia/Shanghai
```

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--interval` | `1.0` | Serving metrics polling interval in seconds |
| `--timeout` | `2.0` | Per-poll HTTP timeout in seconds |
| `--retention-hours` | `24` | Detailed time-series retention |
| `--timezone` | `local` | Timezone used for daily totals |

PyPTO Serving exposes two metrics representations:

- `/metrics` is Prometheus-compatible text for standard collectors.
- `/metrics/json` is the versioned structured interface used by this tool.

The monitor records only aggregate operational metrics. It does not store
prompts, generated text, request IDs, or API credentials.
