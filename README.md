<!-- <h1><img src="assets/logo.png" width="120" alt="deepseek-cursor-proxy logo" style="vertical-align: middle;">&nbsp;DeepSeek Cursor Proxy</h1> -->
<h1 align="center"><img src="assets/logo.png" width="150" alt="deepseek-cursor-proxy logo"><br>DeepSeek Cursor Proxy (async version)</h1>

> **Fork notice:** This is a fork of [deepseek-cursor-proxy](https://github.com/yxlao/deepseek-cursor-proxy) by Yixing Lao, focused on performance, concurrency, and operational improvements. See [original README](https://github.com/yxlao/deepseek-cursor-proxy) for full usage instructions, architecture, and background.

A compatibility proxy that connects Cursor to DeepSeek thinking models (`deepseek-v4-pro` and `deepseek-v4-flash`) by properly handling the `reasoning_content` field for DeepSeek tool-call reasoning API requests. Also works with other applications that hit the same missing `reasoning_content` issue.

## Key Changes from the Original

### Async HTTP (aiohttp)
Migrated from `ThreadingHTTPServer` + `urllib.request` to `aiohttp`. Enables hundreds of concurrent connections in a single event loop, compared to the OS-thread-per-connection model of the original.

### Optimized JSON (orjson)
Replaced standard `json` with `orjson` for ~10x faster serialization and ~3x faster deserialization, reducing per-SSE-chunk processing time during streaming.

### SQLite WAL Mode + Batched Writes
Enabled WAL (Write-Ahead Logging) with `synchronous=NORMAL` and `busy_timeout=5000` for concurrent read/write. Writes are batched (up to 100 rows per transaction); pruning is deferred to every 500 writes instead of after every insert.

### Context Management
New configuration options to control context sent upstream:
- `max_context_messages` — limit the number of messages
- `max_context_tokens` — token-based truncation
- `trim_reasoning_content` / `max_reasoning_chars` — cap reasoning content length

### Diagnostics & Benchmarking
Added `scripts/diagnose.py` — collects system info, benchmarks JSON/regex/SQLite hotspots, runs cProfile, and times the test suite. Requires optional dependencies for full functionality:

```bash
pip install -e ".[diagnose]"
```

### Additional Features
- **Docker support** — `Dockerfile` and `docker-compose.yml` for containerized deployment.
- **CORS support** — optional CORS headers for non-Cursor clients.
- **Request tracing** — optional JSON trace dumps for debugging.
- **Improved config** — additional CLI flags (`--no-ngrok`, `--ngrok-url`, `--verbose`, `--trace-dir`, `--clear-reasoning-cache`, etc.).

## Performance Benchmarks

Results from `scripts/diagnose.py` (reference hardware, n=10,000 iterations):

| Benchmark | Standard | Optimized | Speedup |
|-----------|----------|-----------|---------|
| JSON dumps | 0.007 ms/op | 0.001 ms/op (orjson) | ~10x |
| JSON loads | 0.005 ms/op | 0.001 ms/op (orjson) | ~3x |
| Regex (w/ thinking block) | 0.013 ms/op | — | — |
| SHA-256 hashing | 0.002 ms/op | — | — |
| SQLite INSERT (per-row, no WAL) | 0.003 ms/op | — | baseline |
| SQLite INSERT (per-row, WAL) | — | 0.003 ms/op | ~1x |
| SQLite INSERT (batched, WAL) | — | 0.003 ms/op | ~1.1x |

The test suite completes in ~12 seconds across 97 tests.

## Quick Start

See the [original README](https://github.com/yxlao/deepseek-cursor-proxy) for detailed setup (ngrok, Cursor configuration, first-run behaviour).

```bash
# Clone and install
git clone https://github.com/dimmeagle/deepseek-cursor-proxy-async.git
cd deepseek-cursor-proxy-async
pip install -e .

# Start (with ngrok, or --no-ngrok for local testing)
deepseek-cursor-proxy
```

> **API key resolution:** The proxy resolves the upstream DeepSeek API key in the following order:
> 1. **`DEEPSEEK_API_KEY` environment variable** — if set, this key is used for **all** upstream requests regardless of what the client sends. Useful for shared proxy deployments.
> 2. **Incoming `Authorization` header** — if `DEEPSEEK_API_KEY` is not set, the proxy forwards each client's API key as-is. This allows multiple clients to use their own DeepSeek keys through a single proxy instance.

## Docker Setup

```bash
# Start with docker compose
export DEEPSEEK_API_KEY=sk-your-key-here
docker compose up -d
```

The proxy is available at `http://127.0.0.1:9088` (configurable via `DEEPSEEK_PROXY_PORT` and `DEEPSEEK_PROXY_IP`).

Build manually:

```bash
docker build -t deepseek-cursor-proxy-async -f docker/Dockerfile .
docker run -d \
  --name deepseek-cursor-proxy-async \
  -e DEEPSEEK_API_KEY=sk-your-key-here \
  -p 127.0.0.1:9088:9000 \
  deepseek-cursor-proxy-async
```

The image runs with `--host 0.0.0.0 --port 9000 --no-ngrok` by default.

## Configuration Differences from the Original

This fork adds the following configuration fields beyond what the original supports:

| Field | Default | Description |
|-------|---------|-------------|
| `cors` | `false` | Enable CORS headers for non-Cursor clients |
| `missing_reasoning_strategy` | `recover` | `recover` — patch missing reasoning; `reject` — return 409 |
| `max_context_messages` | `0` | Max messages to send upstream (0 = unlimited) |
| `max_context_tokens` | `0` | Token-based context cap (0 = unlimited) |
| `trim_reasoning_content` | `false` | Cap reasoning content length |
| `max_reasoning_chars` | `5000` | Max reasoning chars (when `trim_reasoning_content` is true) |

## Development

### Running Tests

```bash
uv run python -m unittest discover -s tests

# With pytest
uv run python -m pytest -v

# Live integration test (requires DeepSeek API key)
RUN_LIVE_DEEPSEEK_TESTS=1 LIVE_DEEPSEEK_KEY=sk-your-key uv run python -m unittest tests.test_live
```

97 tests across 9 files. The live test (`test_live.py`) is skipped by default.

### Diagnostics & Benchmarking

```bash
# Quick profile (no server required)
python scripts/diagnose.py

# Full diagnostics
python scripts/diagnose.py --all

# cProfile of hot functions
python scripts/diagnose.py --cprofile

# Benchmark a running proxy
python scripts/diagnose.py --benchmark http://127.0.0.1:9000 --requests 30
```

### Pre-commit Hooks

```bash
uv sync --dev
uv run pre-commit run --all-files
```

## License

MIT
