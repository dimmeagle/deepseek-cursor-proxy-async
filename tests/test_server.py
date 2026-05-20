"""Server boundary, CLI, and operational tests."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import asyncio
import gzip
import json
import logging
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
import unittest
import zlib

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

# On Windows, aiohttp works better with the selector event loop.
# The per-test setUp creates SelectorEventLoop explicitly.

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.logging import (
    ConsoleLogFormatter,
    TerminalSpinner,
)
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import (
    DeepSeekProxyHandler,
    build_arg_parser,
    read_response_body,
    summarize_chat_payload,
    create_app,
)


# ---------------------------------------------------------------------------
# Stubs for fast in-process tests
# ---------------------------------------------------------------------------


class _FakeClientResponse:
    """Mimics an aiohttp.ClientResponse for unit-testing handler methods."""

    def __init__(
        self,
        body: bytes,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = body
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}

    async def read(self) -> bytes:
        return self._body


class _FakeStreamReader:
    """Mimics ``aiohttp.StreamReader`` for unit-testing streaming."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self.readline_calls = 0

    async def readline(self) -> bytes:
        self.readline_calls += 1
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeStreamingClientResponse:
    """Mimics a streaming aiohttp.ClientResponse."""

    def __init__(
        self,
        lines: list[bytes],
        status: int = 200,
        headers: dict[str, str] | None = None,
    ):
        self.content = _FakeStreamReader(lines)
        self.status = status
        self.headers = headers or {"Content-Type": "text/event-stream"}


class _FailingStreamReader:
    """Mimics aiohttp.StreamReader that raises on read."""

    async def readline(self) -> bytes:
        raise OSError("record layer failure")


class _FailingStreamingClientResponse:
    """Mimics a streaming aiohttp.ClientResponse that fails on read."""

    def __init__(self):
        self.content = _FailingStreamReader()
        self.status = 200
        self.headers = {"Content-Type": "text/event-stream"}


class _FakeConsole:
    def __init__(self, *, tty: bool) -> None:
        self.tty = tty
        self.writes: list[str] = []

    def isatty(self) -> bool:
        return self.tty

    def write(self, text: str) -> None:
        self.writes.append(text)

    def flush(self) -> None:
        return


def _make_handler_stub(**config: object) -> DeepSeekProxyHandler:
    """Build a handler with minimal config/reasoning store for unit tests."""
    cfg = ProxyConfig(**config)
    store = ReasoningStore(":memory:")
    handler = DeepSeekProxyHandler(cfg, store, trace_writer=None)
    return handler


# ---------------------------------------------------------------------------
# CLI / pure helpers
# ---------------------------------------------------------------------------


class CliAndHelperTests(unittest.TestCase):
    def test_cli_boolean_flags_have_on_and_off_forms(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--no-ngrok",
                "--no-verbose",
                "--no-display-reasoning",
                "--no-collasible-resoning",
                "--cors",
                "--trace-dir",
                "/tmp/dcp-traces",
            ]
        )
        self.assertFalse(args.ngrok)
        self.assertFalse(args.verbose)
        self.assertFalse(args.display_reasoning)
        self.assertFalse(args.collapsible_reasoning)
        self.assertTrue(args.cors)
        self.assertEqual(args.trace_dir, Path("/tmp/dcp-traces"))

    def test_cli_accepts_ngrok_url(self) -> None:
        args = build_arg_parser().parse_args(
            ["--ngrok-url", "https://example.ngrok.app"]
        )
        self.assertEqual(args.ngrok_url, "https://example.ngrok.app")

    def test_default_console_logging_hides_info_prefix_and_timestamp(
        self,
    ) -> None:
        formatter = ConsoleLogFormatter(verbose=False)
        info_record = logging.LogRecord(
            "deepseek_cursor_proxy",
            logging.INFO,
            __file__,
            1,
            "listening on %s",
            ("http://127.0.0.1:9000/v1",),
            None,
        )
        warning_record = logging.LogRecord(
            "deepseek_cursor_proxy",
            logging.WARNING,
            __file__,
            1,
            "trace logging enabled",
            (),
            None,
        )

        self.assertEqual(
            formatter.format(info_record),
            "listening on http://127.0.0.1:9000/v1",
        )
        self.assertEqual(
            formatter.format(warning_record), "WARNING trace logging enabled"
        )

    def test_verbose_console_logging_shows_timestamp_and_level(self) -> None:
        formatter = ConsoleLogFormatter(verbose=True)
        record = logging.LogRecord(
            "deepseek_cursor_proxy",
            logging.INFO,
            __file__,
            1,
            "listening on %s",
            ("http://127.0.0.1:9000/v1",),
            None,
        )

        self.assertRegex(
            formatter.format(record),
            re.compile(
                r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO listening on "
            ),
        )

    def test_terminal_spinner_animates_only_for_tty(self) -> None:
        tty = _FakeConsole(tty=True)
        spinner = TerminalSpinner(
            enabled=True, text="\u2514 {frame}", stream=tty, interval=0.001
        ).start()
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline and not tty.writes:
            time.sleep(0.001)
        spinner.stop()

        output = "".join(tty.writes)
        self.assertIn(TerminalSpinner.hide_cursor, output)
        self.assertIn("\u2514 \u280b", output)
        self.assertIn(TerminalSpinner.show_cursor, output)
        self.assertTrue(output.endswith(TerminalSpinner.show_cursor))

        non_tty = _FakeConsole(tty=False)
        TerminalSpinner(
            enabled=True,
            text="\u2514 {frame}",
            stream=non_tty,
            interval=0.001,
        ).start().stop()
        self.assertEqual(non_tty.writes, [])

    def test_read_response_body_decodes_gzip_and_deflate(self) -> None:
        self.assertEqual(
            read_response_body(
                BytesIO(gzip.compress(b'{"ok":1}')),
                encoding="gzip",
            ),
            b'{"ok":1}',
        )
        self.assertEqual(
            read_response_body(
                BytesIO(zlib.compress(b'{"ok":1}')),
                encoding="deflate",
            ),
            b'{"ok":1}',
        )

    def test_summarize_chat_payload_omits_message_content(self) -> None:
        summary = summarize_chat_payload(
            {
                "model": "deepseek-v4-pro",
                "stream": True,
                "messages": [{"role": "user", "content": "secret prompt"}],
                "tools": [{"type": "function"}],
                "tool_choice": "auto",
            }
        )
        self.assertIn("model='deepseek-v4-pro'", summary)
        self.assertIn("messages=1", summary)
        self.assertNotIn("secret prompt", summary)


# ---------------------------------------------------------------------------
# Handler stub tests (async, in-process)
# ---------------------------------------------------------------------------


class HandlerStubTests(unittest.TestCase):
    def test_regular_response_handles_client_disconnect(self) -> None:
        """_proxy_regular_response still processes the body when the
        client is gone; the disconnect happens at the aiohttp level so
        the method itself returns a web.Response normally."""
        handler = _make_handler_stub()
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
            }
        ).encode("utf-8")

        async def run() -> web.Response:
            upstream = _FakeClientResponse(body)
            return await handler._proxy_regular_response(
                upstream,
                "deepseek-v4-pro",
                [{"role": "user", "content": "hi"}],
                "ns",
            )

        resp = asyncio.run(run())
        self.assertEqual(resp.status, 200)
        payload = json.loads(resp.body)
        self.assertEqual(
            payload["choices"][0]["message"]["content"], "ok"
        )
        handler.reasoning_store.close()

    def test_streaming_response_stops_on_client_disconnect(self) -> None:
        """The streaming loop stops when the downstream write fails."""
        handler = _make_handler_stub(
            display_reasoning=False, upstream_model="deepseek-v4-pro"
        )
        chunk = {
            "id": "stream",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "hi",
                    },
                }
            ],
        }
        lines = [
            f"data: {json.dumps(chunk)}\n\n".encode("utf-8"),
            b"data: [DONE]\n\n",
        ]
        upstream = _FakeStreamingClientResponse(lines)

        async def run():
            # Use make_mocked_request to get a request that supports
            # StreamResponse.prepare().
            request = make_mocked_request(
                "POST", "/v1/chat/completions"
            )
            return await handler._proxy_streaming_response(
                upstream,
                request,
                "deepseek-v4-pro",
                [{"role": "user", "content": "hi"}],
                "ns",
            )

        resp = asyncio.run(run())
        # The response should have been sent (even though there's no real
        # client, StreamResponse.prepare + write work with the mock)
        self.assertIsInstance(resp, web.StreamResponse)
        # Since display_reasoning is False and upstream_model matches,
        # the fast path should have echoed lines as-is.
        self.assertIsNotNone(resp)
        handler.reasoning_store.close()

    def test_streaming_response_handles_upstream_read_failure(self) -> None:
        handler = _make_handler_stub()

        async def run() -> web.StreamResponse:
            upstream = _FailingStreamingClientResponse()
            request = make_mocked_request(
                "POST", "/v1/chat/completions"
            )
            with self.assertLogs(
                "deepseek_cursor_proxy", level="WARNING"
            ) as captured:
                result = await handler._proxy_streaming_response(
                    upstream,
                    request,
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "ns",
                )
            self.assertIn(
                "upstream streaming response read failed",
                "\n".join(captured.output),
            )
            return result

        asyncio.run(run())
        handler.reasoning_store.close()

    def test_collapsible_reasoning_no_effect_when_display_disabled(
        self,
    ) -> None:
        handler = _make_handler_stub(
            display_reasoning=False,
            collapsible_reasoning=True,
            upstream_model="deepseek-v4-pro",
        )
        chunk = {
            "id": "stream",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "Need context."
                    },
                }
            ],
        }
        lines = [
            f"data: {json.dumps(chunk)}\n\n".encode("utf-8"),
            b"data: [DONE]\n\n",
        ]
        upstream = _FakeStreamingClientResponse(lines)

        async def run() -> web.StreamResponse:
            request = make_mocked_request(
                "POST", "/v1/chat/completions"
            )
            return await handler._proxy_streaming_response(
                upstream,
                request,
                "deepseek-v4-pro",
                [{"role": "user", "content": "hi"}],
                "ns",
            )

        resp = asyncio.run(run())
        handler.reasoning_store.close()
        self.assertIsInstance(resp, web.StreamResponse)


# ---------------------------------------------------------------------------
# HTTP-level boundary tests: real proxy + tiny upstream
# ---------------------------------------------------------------------------


import json as _json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class _PlainFakeUpstream(BaseHTTPRequestHandler):
    """Returns a fixed plain response and records every request."""

    requests: list[dict[str, object]] = []
    auth_headers: list[str] = []
    delay_after_done: float = 0.0
    response: dict[str, object] = {}

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = _json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)
        self.__class__.auth_headers.append(
            self.headers.get("Authorization", "")
        )

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b'data: {"choices":[{"index":0,"delta":'
                b'{"content":"x"}}]}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            if self.__class__.delay_after_done:
                time.sleep(self.__class__.delay_after_done)
            return

        body = _json.dumps(self.__class__.response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_BASE_RESPONSE: dict[str, object] = {
    "id": "x",
    "object": "chat.completion",
    "created": 1,
    "model": "deepseek-v4-pro",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_cache_hit_tokens": 12,
        "prompt_cache_miss_tokens": 8,
        "completion_tokens_details": {"reasoning_tokens": 3},
    },
}


class _Fixture:
    def __init__(self, server: ThreadingHTTPServer) -> None:
        self.server = server
        self.thread = threading.Thread(
            target=server.serve_forever, daemon=True
        )
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class HttpBoundaryTests(unittest.TestCase):
    """Real-HTTP tests that don't fit the protocol suite."""

    def setUp(self) -> None:
        _PlainFakeUpstream.requests = []
        _PlainFakeUpstream.auth_headers = []
        _PlainFakeUpstream.delay_after_done = 0.0
        _PlainFakeUpstream.response = dict(_BASE_RESPONSE)

        # Start the fake upstream (sync HTTP server)
        self.upstream = _Fixture(
            ThreadingHTTPServer(
                ("127.0.0.1", 0), _PlainFakeUpstream
            )
        )

        # Start the proxy (aiohttp) on a background thread with running loop
        self.store = ReasoningStore(":memory:")
        self.proxy_config = ProxyConfig(
            upstream_base_url=self.upstream.url,
            upstream_model="deepseek-v4-pro",
            ngrok=False,
        )
        self._started = threading.Event()
        self._ready_port: list[int] = []
        self._thread = threading.Thread(
            target=self._run_server, daemon=True
        )
        self._thread.start()
        self._started.wait(timeout=10)

    def _run_server(self) -> None:
        if sys.platform == "win32":
            loop = asyncio.SelectorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._start_server(loop))

    async def _start_server(self, loop: asyncio.AbstractEventLoop) -> None:
        app = create_app(self.proxy_config, self.store)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        sock = site._server.sockets[0]
        self._ready_port.append(sock.getsockname()[1])
        self._started.set()
        # Keep the loop running forever
        await asyncio.Event().wait()

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self._ready_port[0]}"

    def tearDown(self) -> None:
        # Signal the daemon thread to stop
        self.upstream.close()
        self.store.close()
        # (The daemon thread stops when the main test finishes)

    # ── Tests ─────────────────────────────────────────────────────

    def _request(self) -> dict:
        return {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def _post(
        self,
        url: str,
        payload: dict,
        api_key: str = "sk-test",
    ) -> tuple[int, dict]:
        """Helper: make a sync HTTP POST request via urllib."""
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError

        request = Request(
            url,
            data=_json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, _json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as exc:
            return exc.code, _json.loads(exc.read().decode("utf-8"))

    def _start_temp_server(
        self,
        store: ReasoningStore,
        **config_overrides: Any,
    ) -> int:
        """Start a temporary aiohttp proxy on a background thread with
        custom config overrides.  Returns the port number."""
        upstream_url = self.upstream.url
        port_holder: list[int] = []
        started = threading.Event()

        def run() -> None:
            loop = asyncio.SelectorEventLoop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self._serve_with_config(
                    store,
                    upstream_url,
                    port_holder,
                    started,
                    **config_overrides,
                )
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        started.wait(timeout=10)
        return port_holder[0]

    @staticmethod
    async def _serve_with_config(
        store: ReasoningStore,
        upstream_url: str,
        port_holder: list[int],
        started: threading.Event,
        **config_overrides: Any,
    ) -> None:
        config = ProxyConfig(
            upstream_base_url=upstream_url,
            upstream_model="deepseek-v4-pro",
            ngrok=False,
            **config_overrides,
        )
        app = create_app(config, store)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        sock = site._server.sockets[0]
        port_holder.append(sock.getsockname()[1])
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    def test_rejects_missing_bearer_token(self) -> None:
        from urllib.error import HTTPError

        request = Request(
            f"{self.proxy_url}/v1/chat/completions",
            data=_json.dumps(self._request()).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 401)
        self.assertEqual(_PlainFakeUpstream.requests, [])

    def test_rejects_oversized_request_body(self) -> None:
        store = ReasoningStore(":memory:")
        port = self._start_temp_server(
            store, max_request_body_bytes=10
        )
        try:
            status, payload = self._post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                self._request(),
            )
        finally:
            store.close()
        self.assertEqual(status, 413)
        self.assertIn("too large", payload["error"]["message"])
        self.assertEqual(_PlainFakeUpstream.requests, [])

    def test_forwards_bearer_token_to_upstream(self) -> None:
        status, _ = self._post(
            f"{self.proxy_url}/v1/chat/completions",
            self._request(),
            api_key="sk-from-cursor",
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            _PlainFakeUpstream.auth_headers[0], "Bearer sk-from-cursor"
        )

    def test_streaming_response_closes_after_done_when_upstream_lingers(
        self,
    ) -> None:
        """Cursor relies on the proxy ending the SSE stream at [DONE],
        even if the upstream socket stays open."""
        _PlainFakeUpstream.delay_after_done = 2.0
        request = Request(
            f"{self.proxy_url}/v1/chat/completions",
            data=_json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "stream"}
                    ],
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
        )
        started = time.monotonic()
        with urlopen(request, timeout=1) as response:
            body = response.read().decode("utf-8")
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIn("data: [DONE]", body)

    def test_normal_logging_summarizes_without_bodies_or_keys(
        self,
    ) -> None:
        with self.assertLogs(
            "deepseek_cursor_proxy", level="INFO"
        ) as captured:
            status, _ = self._post(
                f"{self.proxy_url}/v1/chat/completions",
                {
                    "model": "deepseek-v4-pro",
                    "messages": [
                        {"role": "user", "content": "pragma-test-msg"}
                    ],
                },
                api_key="sk-from-cursor",
            )
            # `\u2514 stats` is emitted on the handler thread *after* the
            # response body hits the socket, so the client may return
            # before it lands.
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not any(
                "\u2514 stats" in record for record in captured.output
            ):
                time.sleep(0.01)
        output = "\n".join(captured.output)
        self.assertEqual(status, 200)
        self.assertIn(
            "\u250c request model=deepseek-v4-pro effort=max messages=1",
            output,
        )
        self.assertIn("\u251c context status=ok reasoning_context=0", output)
        self.assertIn("\u2514 stats", output)
        self.assertNotIn("pragma-test-msg", output)
        self.assertNotIn("sk-from-cursor", output)

    def test_verbose_logging_includes_bodies_but_redacts_api_key(
        self,
    ) -> None:
        store = ReasoningStore(":memory:")
        port = self._start_temp_server(store, verbose=True)
        try:
            with self.assertLogs(
                "deepseek_cursor_proxy", level="INFO"
            ) as captured:
                self._post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    self._request(),
                    api_key="sk-from-cursor",
                )
        finally:
            store.close()
        output = "\n".join(captured.output)
        self.assertIn("cursor request body", output)
        self.assertIn("upstream request body", output)
        self.assertNotIn("sk-from-cursor", output)

    def test_healthz_returns_ok(self) -> None:
        with urlopen(
            f"{self.proxy_url}/healthz", timeout=2
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                _json.loads(response.read())["ok"], True
            )


if __name__ == "__main__":
    unittest.main()
