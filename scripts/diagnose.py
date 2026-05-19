#!/usr/bin/env python3
"""Phase 0: Baseline diagnostics for deepseek-cursor-proxy.

Collects system info, benchmarks key hot-functions (JSON, regex, SQLite),
runs the test suite, and optionally benchmarks a running proxy.

Usage:
    # Quick system + hotspot profile (no server needed)
    python scripts/diagnose.py

    # Full diagnostics
    python scripts/diagnose.py --all

    # Benchmark a running proxy (replace URL with yours)
    python scripts/diagnose.py --benchmark http://127.0.0.1:9000

    # Profile with cProfile and generate stats
    python scripts/diagnose.py --cprofile
"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import os
import pstats
import platform
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ── ANSI helpers ──────────────────────────────────────────────────────────────
# Detect if terminal supports Unicode; fall back to ASCII on Windows codepages
# that don't support the characters we use (cp1251, cp866, etc).

def _can_unicode() -> bool:
    try:
        # Try to encode a representative Unicode character
        "\u2550\u2713\u26a0\u2717".encode(sys.stdout.encoding or "utf-8", errors="strict")
        return True
    except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
        return False


_UNICODE = _can_unicode()


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if sys.stderr.isatty() else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if sys.stderr.isatty() else text


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if sys.stderr.isatty() else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if sys.stderr.isatty() else text


def _hline(char: str = "=") -> str:
    return char * 60


def _header(text: str) -> str:
    h = _hline("=")
    return f"\n{_bold(h)}\n{_bold(f'  {text}')}\n{_bold(h)}"


def _subheader(text: str) -> str:
    return f"\n  {_bold(text)}"


_OK = _green("OK") if not _UNICODE else _green("✓")
_WARN_SIGN = _yellow("!!") if not _UNICODE else _yellow(" ⚠ ")
_FAIL_SIGN = _red("XX") if not _UNICODE else _red(" ✗ ")


def _ok() -> str:
    return _OK


def _warn(text: str) -> str:
    return f"  {_WARN_SIGN}  {text}"


def _fail(text: str) -> str:
    return f"  {_FAIL_SIGN}  {text}"


# ── Section 1: System Info ────────────────────────────────────────────────────


def section_system() -> dict[str, Any]:
    print(_header("1. System Information"))

    info: dict[str, Any] = {}

    py_impl = platform.python_implementation()
    py_version = platform.python_version()
    py_compiler = platform.python_compiler()
    print(f"  Python:      {py_impl} {py_version} ({py_compiler})")
    info["python_implementation"] = py_impl
    info["python_version"] = py_version
    info["python_compiler"] = py_compiler

    os_name = platform.system()
    os_release = platform.release()
    print(f"  OS:          {os_name} {os_release}")
    info["os"] = os_name
    info["os_release"] = os_release

    machine = platform.machine()
    processor = platform.processor() or "?"
    print(f"  Machine:     {machine} ({processor})")
    info["machine"] = machine
    info["processor"] = processor

    cpus = os.cpu_count() or 0
    print(f"  CPUs:        {cpus}")
    info["cpu_count"] = cpus

    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        avail_gb = mem.available / (1024 ** 3)
        print(f"  RAM:         {total_gb:.1f} GB total, {avail_gb:.1f} GB available "
              f"({mem.percent}% used)")
        info["ram_total_gb"] = round(total_gb, 1)
        info["ram_avail_gb"] = round(avail_gb, 1)
        info["ram_percent_used"] = mem.percent
    except ImportError:
        print(_warn("psutil not installed — run `pip install psutil` for RAM info"))
        info["ram"] = "unknown (psutil not available)"

    # Python build flags (GIL, etc.)
    import sysconfig
    py_config = sysconfig.get_config_vars()
    gil_enabled = py_config.get("Py_GIL_DISABLED", None)
    if gil_enabled is not None:
        gil_status = f"Py_GIL_DISABLED={gil_enabled}"
    else:
        gil_status = "Py_GIL_ENABLED (default)"
    print(f"  GIL status:  {gil_status}")
    info["gil_status"] = gil_status

    # Check for common optimizations / libs
    libs = {
        "orjson": False,
        "aiohttp": False,
        "psutil": False,
        "uvloop": False,
    }
    for lib_name in libs:
        try:
            __import__(lib_name)
            libs[lib_name] = True
        except ImportError:
            pass
    print(f"  Optimized libs:")
    for lib_name, present in libs.items():
        print(f"    {lib_name}: {_green('installed') if present else _yellow('not installed')}")
    info["libraries"] = libs

    return info


# ── Section 2: Config ────────────────────────────────────────────────────────


def section_config() -> dict[str, Any]:
    print(_header("2. Current Configuration"))

    config_path = Path.home() / ".deepseek-cursor-proxy" / "config.yaml"
    info: dict[str, Any] = {"config_path": str(config_path)}

    if not config_path.exists():
        print(_warn(f"Config not found at {config_path} (using defaults)"))
        info["exists"] = False
        return info

    info["exists"] = True
    content = config_path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    print(f"  Path:    {config_path}")
    print(f"  Size:    {len(content)} bytes, {len(lines)} lines")
    print(f"  Content:")
    for line in lines:
        if "key" in line.lower() or "token" in line.lower() or "password" in line.lower():
            print(_red(f"    {line}  *** WARNING: potential secret in config! ***"))
        else:
            print(f"    {line}")
    info["size_bytes"] = len(content)
    info["line_count"] = len(lines)

    return info


# ── Section 3: SQLite Reasoning Store ────────────────────────────────────────


def section_sqlite() -> dict[str, Any]:
    print(_header("3. SQLite Reasoning Store"))

    info: dict[str, Any] = {}

    db_path = Path.home() / ".deepseek-cursor-proxy" / "reasoning_content.sqlite3"
    if not db_path.exists():
        print(_warn(f"Reasoning store not found at {db_path}"))
        info["exists"] = False
        return info

    info["exists"] = True
    size_bytes = db_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"  Path:      {db_path}")
    print(f"  Size:      {size_mb:.2f} MB ({size_bytes:,} bytes)")
    info["path"] = str(db_path)
    info["size_bytes"] = size_bytes

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode")
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        print(f"  Journal:   {journal_mode}")
        info["journal_mode"] = journal_mode

        row_count = conn.execute(
            "SELECT COUNT(*) FROM reasoning_cache"
        ).fetchone()[0]
        print(f"  Rows:      {row_count:,}")
        info["row_count"] = row_count

        oldest = conn.execute(
            "SELECT MIN(created_at) FROM reasoning_cache"
        ).fetchone()[0]
        newest = conn.execute(
            "SELECT MAX(created_at) FROM reasoning_cache"
        ).fetchone()[0]
        if oldest and newest:
            import datetime
            oldest_dt = datetime.datetime.fromtimestamp(oldest, tz=datetime.timezone.utc)
            newest_dt = datetime.datetime.fromtimestamp(newest, tz=datetime.timezone.utc)
            age_days = (newest_dt - oldest_dt).days
            print(f"  Age range: {oldest_dt.date()} to {newest_dt.date()} "
                  f"({age_days} days)")
            info["oldest"] = oldest_dt.isoformat()
            info["newest"] = newest_dt.isoformat()

        # Sample a few rows for avg reasoning length
        sample = conn.execute(
            "SELECT AVG(LENGTH(reasoning)) FROM reasoning_cache"
        ).fetchone()[0]
        if sample:
            avg_chars = int(sample)
            print(f"  Avg reasoning length: {avg_chars:,} chars")
            info["avg_reasoning_length_chars"] = avg_chars

        # WAL size
        wal_path = db_path.with_suffix(".sqlite3-wal")
        if wal_path.exists():
            wal_size = wal_path.stat().st_size / 1024
            print(f"  WAL size:  {wal_size:.1f} KB")
            info["wal_size_kb"] = round(wal_size, 1)

        conn.close()
    except sqlite3.Error as exc:
        print(_fail(f"SQLite error: {exc}"))

    return info


# ── Section 4: Hotspot Profiling ─────────────────────────────────────────────


def section_hotspots(count: int = 10000) -> dict[str, Any]:
    print(_header(f"4. Hotspot Benchmarks (n={count:,})"))

    results: dict[str, float] = {}

    # ── 4a: JSON serialization ──
    print(_subheader("4a. JSON dumps (standard json)"))
    sample = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                    "reasoning_content": "The user is asking a general question...",
                },
                "finish_reason": None,
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 12,
            "total_tokens": 162,
        },
    }

    t0 = time.perf_counter()
    for _ in range(count):
        json.dumps(sample, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    t1 = time.perf_counter()
    std_time = t1 - t0
    print(f"    json.dumps + encode:  {std_time*1000/count:.3f} ms/op  "
          f"({std_time:.2f}s total)")
    results["json_std_dumps_ms"] = round(std_time * 1000 / count, 4)

    # ── 4b: orjson if available ──
    try:
        import orjson
        t0 = time.perf_counter()
        for _ in range(count):
            orjson.dumps(sample)
        t1 = time.perf_counter()
        orjson_time = t1 - t0
        speedup = std_time / orjson_time if orjson_time > 0 else 0
        print(f"    orjson.dumps:         {orjson_time*1000/count:.3f} ms/op  "
              f"({orjson_time:.2f}s total)  "
              f"{_green(f'{speedup:.1f}x faster') if speedup > 1 else ''}")
        results["orjson_dumps_ms"] = round(orjson_time * 1000 / count, 4)
        results["json_speedup_orjson"] = round(speedup, 1)
    except ImportError:
        print(_warn("orjson not installed — skipping"))

    # ── 4c: JSON loads ──
    print(_subheader("4c. JSON loads"))
    sample_bytes = json.dumps(sample).encode("utf-8")

    t0 = time.perf_counter()
    for _ in range(count):
        json.loads(sample_bytes.decode("utf-8"))
    t1 = time.perf_counter()
    std_load_time = t1 - t0
    print(f"    json.loads (str):     {std_load_time*1000/count:.3f} ms/op  "
          f"({std_load_time:.2f}s total)")
    results["json_std_loads_ms"] = round(std_load_time * 1000 / count, 4)

    try:
        import orjson
        t0 = time.perf_counter()
        for _ in range(count):
            orjson.loads(sample_bytes)
        t1 = time.perf_counter()
        orjson_load_time = t1 - t0
        speedup = std_load_time / orjson_load_time if orjson_load_time > 0 else 0
        print(f"    orjson.loads (bytes): {orjson_load_time*1000/count:.3f} ms/op  "
              f"({orjson_load_time:.2f}s total)  "
              f"{_green(f'{speedup:.1f}x faster') if speedup > 1 else ''}")
        results["orjson_loads_ms"] = round(orjson_load_time * 1000 / count, 4)
        results["json_loads_speedup_orjson"] = round(speedup, 1)
    except ImportError:
        pass

    # ── 4d: Regex (thinking block stripping) ──
    print(_subheader("4d. Regex (thinking block stripping)"))
    # Simulate content with and without thinking blocks
    short_content = "Hello, this is a short message from the assistant."
    long_content = (
        "Let me think about this step by step.\n\n"
        "<details>\n<summary>Thinking</summary>\n\n"
        "First, I need to analyze the user's request. They are asking about "
        "something that requires deep reasoning. Let me break this down...\n\n"
        "The key insight here is that we need to consider multiple factors. "
        "Let me enumerate them:\n"
        "1. Factor one is important because...\n"
        "2. Factor two changes the analysis...\n"
        "3. Factor three is the deciding factor...\n\n"
        "Therefore, the answer is clear.\n"
        "</details>\n\n"
        "The answer to your question is 42."
    )

    block_pattern = re.compile(
        r"""
        (?:
            <(?:think|thinking)\b[^>]*>[\s\S]*?(?:</(?:think|thinking)>|\Z)
            |
            <details\b[^>]*>\s*
            <summary\b[^>]*>\s*Thinking\s*</summary>
            [\s\S]*?(?:</details>|\Z)
        )\s*
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    t0 = time.perf_counter()
    for _ in range(count):
        block_pattern.sub("", short_content)
    t1 = time.perf_counter()
    short_regex_time = t1 - t0
    print(f"    Short content:          {short_regex_time*1000/count:.3f} ms/op  "
          f"({short_regex_time:.2f}s total)")
    results["regex_short_ms"] = round(short_regex_time * 1000 / count, 4)

    t0 = time.perf_counter()
    for _ in range(count):
        block_pattern.sub("", long_content)
    t1 = time.perf_counter()
    long_regex_time = t1 - t0
    print(f"    Long content (w/ block): {long_regex_time*1000/count:.3f} ms/op  "
          f"({long_regex_time:.2f}s total)")
    results["regex_long_ms"] = round(long_regex_time * 1000 / count, 4)

    # ── 4e: SQLite INSERT (in-memory) ──
    print(_subheader("4e. SQLite INSERT (in-memory)"))
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS perf_test ("
        "key TEXT PRIMARY KEY, value TEXT, created_at REAL)"
    )

    # WAL vs no WAL comparison
    conn_no_wal = sqlite3.connect(":memory:")
    conn_no_wal.execute(
        "CREATE TABLE IF NOT EXISTS perf_test ("
        "key TEXT PRIMARY KEY, value TEXT, created_at REAL)"
    )

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    test_rows = [(f"key:{i}", f"value:{i}" * 10, time.time()) for i in range(count)]

    # Without WAL
    t0 = time.perf_counter()
    for key, value, ts in test_rows:
        conn_no_wal.execute(
            "INSERT OR REPLACE INTO perf_test VALUES (?, ?, ?)",
            (key, value, ts),
        )
    conn_no_wal.commit()
    t1 = time.perf_counter()
    no_wal_time = t1 - t0
    print(f"    Without WAL (per-row):   {no_wal_time*1000/count:.3f} ms/op  "
          f"({no_wal_time:.2f}s total)")
    results["sqlite_no_wal_per_row_ms"] = round(no_wal_time * 1000 / count, 4)

    # With WAL
    t0 = time.perf_counter()
    for key, value, ts in test_rows:
        conn.execute(
            "INSERT OR REPLACE INTO perf_test VALUES (?, ?, ?)",
            (key, value, ts),
        )
    conn.commit()
    t1 = time.perf_counter()
    wal_time = t1 - t0
    print(f"    With WAL (per-row):      {wal_time*1000/count:.3f} ms/op  "
          f"({wal_time:.2f}s total)  "
          f"{_green(f'{no_wal_time/wal_time:.1f}x faster') if wal_time > 0 else ''}")
    results["sqlite_wal_per_row_ms"] = round(wal_time * 1000 / count, 4)

    # Batched INSERT (executemany) with WAL
    t0 = time.perf_counter()
    conn.executemany(
        "INSERT OR REPLACE INTO perf_test VALUES (?, ?, ?)",
        test_rows,
    )
    conn.commit()
    t1 = time.perf_counter()
    batch_time = t1 - t0
    print(f"    With WAL (executemany):  {batch_time*1000/count:.3f} ms/op  "
          f"({batch_time:.2f}s total)  "
          f"{_green(f'{no_wal_time/batch_time:.1f}x faster') if batch_time > 0 else ''}")
    results["sqlite_wal_batch_ms"] = round(batch_time * 1000 / count, 4)

    conn.close()
    conn_no_wal.close()

    # ── 4f: SHA-256 hashing ──
    print(_subheader("4f. SHA-256 hashing"))
    test_string = json.dumps(sample, sort_keys=True)
    import hashlib
    t0 = time.perf_counter()
    for _ in range(count):
        hashlib.sha256(test_string.encode("utf-8")).hexdigest()
    t1 = time.perf_counter()
    hash_time = t1 - t0
    print(f"    SHA-256:                 {hash_time*1000/count:.3f} ms/op  "
          f"({hash_time:.2f}s total)")
    results["sha256_ms"] = round(hash_time * 1000 / count, 4)

    return results


# ── Section 5: Test Suite Timing ─────────────────────────────────────────────


def section_test_timing() -> dict[str, Any]:
    print(_header("5. Test Suite Timing"))

    results: dict[str, Any] = {}
    test_dir = Path(__file__).resolve().parent.parent / "tests"

    if not test_dir.exists():
        print(_warn(f"No tests directory at {test_dir}"))
        return results

    print(f"  Test directory: {test_dir}")
    results["test_dir"] = str(test_dir)

    test_files = sorted(test_dir.glob("test_*.py"))
    print(f"  Found {len(test_files)} test file(s):")
    for tf in test_files:
        print(f"    - {tf.name}")

    print(f"\n  Running tests (this may take a while)...")
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(test_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.monotonic() - start

    print(f"  {'─' * 50}")
    for line in result.stdout.splitlines():
        if any(keyword in line for keyword in ("FAIL", "ERROR", "FAILED", "ERRORS")):
            print(f"  {_red(line)}")
        elif "OK" in line:
            print(f"  {_green(line)}")
        elif "test_" in line:
            if line.strip().startswith("test_") or " ... " in line:
                if "ok" in line.lower() or "pass" in line.lower():
                    print(f"    {_ok()} {line}")
                elif "FAIL" in line or "ERROR" in line:
                    ok_sign = _FAIL_SIGN
                    # Show inline with test line
                    extract = line.split(" ... ")[0] if " ... " in line else line.strip()
                    print(f"    {_fail('')} {extract}")
                else:
                    print(f"    {line}")
            else:
                print(f"  {line}")
        else:
            print(f"  {line}")

    print(f"  {'─' * 50}")
    print(f"  Total test time: {elapsed:.1f}s")
    results["elapsed_seconds"] = round(elapsed, 1)
    results["return_code"] = result.returncode
    results["passed"] = result.returncode == 0

    if result.stderr.strip():
        for line in result.stderr.splitlines():
            print(f"  {_yellow(line)}")

    return results


# ── Section 6: Live Benchmark ────────────────────────────────────────────────


def section_benchmark(
    base_url: str,
    num_requests: int = 20,
    parallel: int = 1,
) -> dict[str, Any]:
    print(_header(f"6. Live Proxy Benchmark"))
    print(f"  Target:  {base_url}")
    print(f"  Requests: {num_requests}")
    print(f"  Parallel: {parallel}")

    results: dict[str, Any] = {
        "target": base_url,
        "num_requests": num_requests,
    }

    # Test health endpoint first
    for endpoint in ["/healthz", "/v1/healthz", "/models", "/v1/models"]:
        url = f"{base_url.rstrip('/v1')}{endpoint}"
        try:
            t0 = time.perf_counter()
            with urlopen(Request(url), timeout=5) as resp:
                body = resp.read()
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  {_ok()} GET {endpoint}: {elapsed:.0f}ms (status {resp.status})")
            results[f"get_{endpoint.replace('/', '_')}_ms"] = round(elapsed, 0)
        except (HTTPError, URLError, OSError) as exc:
            print(f"  {_warn(f'GET {endpoint}: {exc}')}")

    # Benchmark POST /chat/completions (non-streaming)
    test_payload = {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say OK."},
        ],
        "max_tokens": 10,
        "stream": False,
    }

    latencies: list[float] = []
    print(f"\n  POST /v1/chat/completions ({num_requests} requests)...")
    # Warmup
    try:
        _proxy_request(base_url, test_payload, timeout=60)
    except Exception:
        pass

    for i in range(num_requests):
        t0 = time.perf_counter()
        try:
            _proxy_request(base_url, test_payload, timeout=120)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
        except Exception as exc:
            print(f"  {_fail(f'request {i+1}: {exc}')}")

    if latencies:
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        mean = statistics.mean(latencies)
        stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        print(f"\n  {'─' * 50}")
        print(f"  Latency Statistics ({len(latencies)} samples):")
        print(f"    Mean:   {mean:.0f} ms")
        print(f"    StdDev: {stdev:.0f} ms")
        print(f"    Min:    {min(latencies):.0f} ms")
        print(f"    P50:    {p50:.0f} ms")
        print(f"    P95:    {p95:.0f} ms")
        print(f"    P99:    {p99:.0f} ms")
        print(f"    Max:    {max(latencies):.0f} ms")
        results["latency_ms"] = {
            "mean": round(mean, 0),
            "stddev": round(stdev, 0),
            "min": round(min(latencies), 0),
            "p50": round(p50, 0),
            "p95": round(p95, 0),
            "p99": round(p99, 0),
            "max": round(max(latencies), 0),
        }
    else:
        print(f"  {_fail('All requests failed - is the proxy running?')}")

    # Benchmark streaming
    print(f"\n  POST /v1/chat/completions (streaming, {min(num_requests, 5)} requests)...")
    stream_payload = dict(test_payload, stream=True)
    stream_latencies: list[float] = []
    for i in range(min(num_requests, 5)):
        t0 = time.perf_counter()
        try:
            _proxy_stream_request(base_url, stream_payload, timeout=120)
            elapsed = (time.perf_counter() - t0) * 1000
            stream_latencies.append(elapsed)
        except Exception as exc:
            print(f"  {_fail(f'stream {i+1}: {exc}')}")

    if stream_latencies:
        print(f"    Mean TTFB+total: {statistics.mean(stream_latencies):.0f} ms")
        results["stream_latency_ms"] = {
            "mean": round(statistics.mean(stream_latencies), 0),
        }

    return results


def _proxy_request(base_url: str, payload: dict, timeout: int = 60) -> bytes:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    if "/v1/v1/" in url:
        url = url.replace("/v1/v1/", "/v1/")
    if not url.endswith("/chat/completions") and "/chat/completions" not in url:
        # Try adding /v1 prefix
        url = f"{base_url.rstrip('/')}/v1/chat/completions"

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer sk-diagnose",
            "Content-Type": "application/json",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _proxy_stream_request(base_url: str, payload: dict, timeout: int = 120) -> list[bytes]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    if "/v1/v1/" in url:
        url = url.replace("/v1/v1/", "/v1/")
    if not url.endswith("/chat/completions") and "/chat/completions" not in url:
        url = f"{base_url.rstrip('/')}/v1/chat/completions"

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer sk-diagnose",
            "Content-Type": "application/json",
        },
    )
    chunks: list[bytes] = []
    with urlopen(req, timeout=timeout) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            chunks.append(line)
            if b"[DONE]" in line:
                break
    return chunks


# ── Section 7: cProfile ──────────────────────────────────────────────────────


def section_cprofile(duration: int = 10) -> dict[str, Any]:
    print(_header("7. cProfile — Local Hot Functions"))

    results: dict[str, Any] = {}

    # Profile the hotspot functions locally
    profiler = cProfile.Profile()
    profiler.enable()

    # Run a synthetic workload
    import hashlib
    sample = {"a": "b" * 100, "c": [1, 2, 3] * 50}
    for _ in range(5000):
        json.dumps(sample, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        hashlib.sha256(json.dumps(sample, sort_keys=True).encode("utf-8")).hexdigest()

    # Regex
    pat = re.compile(r"<(?:think|thinking)\b[^>]*>.*?</(?:think|thinking)>", re.DOTALL)
    text = "Some text <think>deep thoughts</think> more text" * 200
    for _ in range(5000):
        pat.sub("", text)

    profiler.disable()

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
    stats.print_stats(20)

    print(f"  Top 20 functions by cumulative time:")
    print(f"  {'─' * 60}")
    output = s.getvalue()
    for line in output.splitlines()[:25]:
        print(f"  {line}")
    results["profile_output"] = output

    return results


# ── Summary ──────────────────────────────────────────────────────────────────


def print_summary(all_results: dict[str, Any]) -> None:
    print(_header("SUMMARY"))
    print(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    if "system" in all_results:
        print(f"  Python: {all_results['system'].get('python_version', '?')}")
        print(f"  OS:     {all_results['system'].get('os', '?')} "
              f"{all_results['system'].get('os_release', '?')}")

    if "hotspots" in all_results:
        hs = all_results["hotspots"]
        print(f"\n  Key benchmarks:")
        if "json_std_dumps_ms" in hs:
            std = hs["json_std_dumps_ms"]
            print(f"    json.dumps:    {std:.4f} ms/op")
        if "orjson_dumps_ms" in hs:
            orj = hs["orjson_dumps_ms"]
            print(f"    orjson.dumps:  {orj:.4f} ms/op  "
                  f"({hs.get('json_speedup_orjson', '?')}x faster)")
        if "regex_long_ms" in hs:
            print(f"    Regex (long):  {hs['regex_long_ms']:.4f} ms/op")
        if "sqlite_wal_batch_ms" in hs:
            print(f"    SQLite batch:  {hs['sqlite_wal_batch_ms']:.4f} ms/op")

    if "benchmark" in all_results and "latency_ms" in all_results["benchmark"]:
        lat = all_results["benchmark"]["latency_ms"]
        print(f"\n  Proxy latency:")
        print(f"    P50: {lat['p50']:.0f} ms  |  P95: {lat['p95']:.0f} ms  |  "
              f"P99: {lat['p99']:.0f} ms")

    if "test_timing" in all_results:
        test = all_results["test_timing"]
        status = _green("PASS") if test.get("passed") else _red("FAIL")
        print(f"\n  Test suite: {status}  ({test.get('elapsed_seconds', '?'):.1f}s)")

    if "sqlite" in all_results:
        sq = all_results["sqlite"]
        if sq.get("exists"):
            print(f"\n  SQLite store: {sq.get('row_count', '?'):,} rows, "
                  f"{sq.get('size_bytes', 0) / 1024 / 1024:.1f} MB")

    print(f"\n  {_green('Diagnostics complete.')}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 0 diagnostics for deepseek-cursor-proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all diagnostics except --benchmark"
    )
    parser.add_argument(
        "--system", action="store_true", help="Collect system information"
    )
    parser.add_argument(
        "--config", action="store_true", help="Show current configuration"
    )
    parser.add_argument(
        "--sqlite", action="store_true", help="Analyze SQLite reasoning store"
    )
    parser.add_argument(
        "--hotspots", type=int, nargs="?",
        const=10000, metavar="N",
        help="Benchmark hot functions (JSON, regex, SQLite, SHA)",
    )
    parser.add_argument(
        "--cprofile", action="store_true", help="Profile hot functions with cProfile"
    )
    parser.add_argument(
        "--tests", action="store_true", help="Run the test suite and measure time"
    )
    parser.add_argument(
        "--benchmark", type=str, nargs="?",
        const="http://127.0.0.1:9000", metavar="URL",
        help="Benchmark a running proxy (default: http://127.0.0.1:9000)",
    )
    parser.add_argument(
        "--requests", type=int, default=20,
        help="Number of benchmark requests (default: 20)",
    )

    args = parser.parse_args()

    # If no args, run system + hotspots by default
    run_all = args.all or not any(
        [args.system, args.config, args.sqlite, args.hotspots is not None,
         args.cprofile, args.tests, args.benchmark is not None]
    )

    all_results: dict[str, Any] = {}

    if run_all or args.system:
        all_results["system"] = section_system()

    if run_all or args.config:
        all_results["config"] = section_config()

    if run_all or args.sqlite:
        all_results["sqlite"] = section_sqlite()

    if run_all or args.hotspots is not None:
        n = args.hotspots if args.hotspots is not None else 10000
        all_results["hotspots"] = section_hotspots(count=n)

    if run_all or args.cprofile:
        all_results["cprofile"] = section_cprofile()

    if run_all or args.tests:
        all_results["test_timing"] = section_test_timing()

    if args.benchmark is not None:
        all_results["benchmark"] = section_benchmark(
            args.benchmark, num_requests=args.requests
        )

    # Print header again, then summary
    print(f"\n{'=' * 60}")
    print_summary(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
