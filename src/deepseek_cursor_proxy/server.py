"""Local OpenAI-compatible proxy for Cursor DeepSeek reasoning models.

Async HTTP server built on aiohttp. Replaced the original
ThreadingHTTPServer + urllib implementation for better concurrency.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
import gzip
import orjson
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlparse
import zlib

from aiohttp import web
from aiohttp.client import ClientResponse
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from .config import (
    ProxyConfig,
    default_config_path,
    default_reasoning_content_path,
)
from .logging import (
    LOG,
    TerminalSpinner,
    configure_logging,
)
from .reasoning_store import ReasoningStore, conversation_scope
from .streaming import CursorReasoningDisplayAdapter, StreamAccumulator
from .trace import TraceRequest, TraceWriter
from .tunnel import NgrokTunnel, local_tunnel_target
from .transform import (
    RECOVERY_NOTICE_CONTENT,
    prepare_upstream_request,
    rewrite_response_body,
)


class RequestBodyTooLarge(ValueError):
    pass


@dataclass
class ProxyResponseResult:
    sent: bool
    usage: dict[str, Any] | None = None


SERVER_VERSION = "DeepSeekPythonProxy/0.1"


# ---------------------------------------------------------------------------
# Handler — one instance per application (shared across requests)
# ---------------------------------------------------------------------------


class DeepSeekProxyHandler:
    """Async HTTP handler for the DeepSeek Cursor proxy.

    Uses a single aiohttp ``ClientSession`` (created in ``start()``) for
    all upstream requests.  The session lives as long as the application.
    """

    def __init__(
        self,
        config: ProxyConfig,
        reasoning_store: ReasoningStore,
        trace_writer: TraceWriter | None,
    ) -> None:
        self.config = config
        self.reasoning_store = reasoning_store
        self.trace_writer = trace_writer
        self._session: ClientSession | None = None

    async def start(self) -> None:
        """Create the shared HTTP client session."""
        connector = TCPConnector(
            limit=100,
            limit_per_host=10,
            enable_cleanup_closed=True,
            force_close=False,  # keep-alive + HTTP/2 multiplexing
        )
        self._session = ClientSession(
            connector=connector,
            timeout=ClientTimeout(total=self.config.request_timeout),
        )

    async def stop(self) -> None:
        """Close the shared HTTP client session."""
        if self._session is not None:
            await self._session.close()

    # ── Route handlers ────────────────────────────────────────────────

    async def healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def models(self, request: web.Request) -> web.Response:
        created = int(time.time())
        model_ids = list(
            dict.fromkeys(
                [
                    self.config.upstream_model,
                    "deepseek-v4-pro",
                    "deepseek-v4-flash",
                ]
            )
        )
        models_list = [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
            }
            for model_id in model_ids
        ]
        return web.json_response({"object": "list", "data": models_list})

    async def chat_completions(
        self, request: web.Request
    ) -> web.StreamResponse | web.Response:
        started = time.monotonic()
        request_path = request.path
        trace = self._start_trace(request_path, request)

        if self.config.verbose:
            LOG.info(
                "incoming POST %s from %s content_length=%s user_agent=%s",
                request_path,
                request.remote,
                request.headers.get("Content-Length", "0"),
                request.headers.get("User-Agent", ""),
            )

        # ── Path check ────────────────────────────────────────────
        if request_path not in {"/chat/completions", "/v1/chat/completions"}:
            LOG.warning(
                "rejected unsupported POST path=%s status=404", request_path
            )
            await self._record_request_body_for_trace(request, trace)
            self._finish_trace(trace, "rejected", http_status=404)
            return web.json_response(
                {"error": {"message": "Only /v1/chat/completions is supported"}},
                status=404,
            )

        # ── Authorization ─────────────────────────────────────────
        cursor_authorization = self._cursor_authorization(request)
        if cursor_authorization is None:
            LOG.warning(
                "rejected request path=%s status=401 reason=missing_bearer_token",
                request_path,
            )
            await self._record_request_body_for_trace(request, trace)
            self._finish_trace(trace, "rejected", http_status=401)
            return web.json_response(
                {"error": {"message": "Missing Authorization bearer token"}},
                status=401,
            )

        # ── Read request body ─────────────────────────────────────
        try:
            payload = await self._read_json_body(request)
        except RequestBodyTooLarge as exc:
            LOG.warning(
                "rejected request path=%s status=413 reason=%s",
                request_path,
                exc,
            )
            self._finish_trace(trace, "rejected", http_status=413, reason=str(exc))
            return web.json_response(
                {"error": {"message": str(exc)}}, status=413
            )
        except ValueError as exc:
            LOG.warning(
                "rejected request path=%s status=400 reason=%s",
                request_path,
                exc,
            )
            self._finish_trace(
                trace, "rejected", http_status=400, reason=str(exc)
            )
            return web.json_response(
                {"error": {"message": str(exc)}}, status=400
            )

        if trace is not None:
            trace.record_cursor_body(payload)

        if self.config.verbose:
            log_json("cursor request body", payload)

        log_cursor_request(payload, self.config)

        # ── Prepare upstream request ──────────────────────────────
        prepared = prepare_upstream_request(
            payload,
            self.config,
            self.reasoning_store,
            authorization=cursor_authorization,
        )
        if trace is not None:
            trace.record_transform(prepared)
        log_context_summary(prepared)

        # ── Reject mode (strict) ──────────────────────────────────
        if (
            prepared.missing_reasoning_messages
            and self.config.missing_reasoning_strategy == "reject"
        ):
            LOG.warning(
                (
                    "strict missing-reasoning mode rejected request path=%s "
                    "status=409 reason=missing_reasoning_content count=%s"
                ),
                request_path,
                prepared.missing_reasoning_messages,
            )
            self._finish_trace(trace, "rejected", http_status=409)
            return web.json_response(
                {
                    "error": {
                        "message": (
                            "deepseek-cursor-proxy is running in strict "
                            "missing-reasoning mode and cannot automatically "
                            "recover this thinking-mode tool-call history because "
                            "cached DeepSeek reasoning_content is missing for "
                            f"{prepared.missing_reasoning_messages} assistant "
                            "message(s). Restart without "
                            "`--missing-reasoning-strategy reject`, or pass "
                            "`--missing-reasoning-strategy recover`, so the proxy "
                            "can recover from partial chat history automatically."
                        ),
                        "type": "missing_reasoning_content",
                        "code": "missing_reasoning_content",
                        "missing_reasoning_messages": prepared.missing_reasoning_messages,
                    }
                },
                status=409,
            )

        # ── Verbose logging ───────────────────────────────────────
        if self.config.verbose:
            LOG.info(
                (
                    "upstream request metadata: original_model=%s "
                    "upstream_model=%s patched_reasoning=%s "
                    "missing_reasoning=%s %s"
                ),
                prepared.original_model,
                prepared.upstream_model,
                prepared.patched_reasoning_messages,
                prepared.missing_reasoning_messages,
                summarize_chat_payload(prepared.payload),
            )
            log_json("upstream request body", prepared.payload)

        # ── Forward to upstream ───────────────────────────────────
        upstream_body = orjson.dumps(prepared.payload)
        upstream_url = f"{self.config.upstream_base_url}/chat/completions"
        upstream_headers = self._upstream_headers(
            stream=bool(prepared.payload.get("stream")),
            authorization=cursor_authorization,
        )
        if trace is not None:
            trace.record_upstream_request(
                url=upstream_url,
                headers=upstream_headers,
                body_bytes=upstream_body,
            )

        log_send_summary(prepared)
        spinner = TerminalSpinner(
            enabled=bool(prepared.payload.get("stream"))
            and not self.config.verbose,
            text="\u2514 {frame}",
        ).start()

        try:
            upstream_resp = await self._session.post(
                upstream_url,
                data=upstream_body,
                headers=upstream_headers,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            spinner.stop()
            LOG.warning(
                "upstream request failed elapsed_ms=%s reason=%s",
                elapsed_ms(started),
                exc,
            )
            self._finish_trace(trace, "upstream_error", http_status=502)
            return web.json_response(
                {
                    "error": {
                        "message": f"Upstream request failed: {exc}"
                    }
                },
                status=502,
            )

        upstream_status = upstream_resp.status
        if upstream_status >= 400:
            spinner.stop()
            LOG.warning(
                "request failed upstream_status=%s stream=%s elapsed_ms=%s",
                upstream_status,
                bool(prepared.payload.get("stream")),
                elapsed_ms(started),
            )
            return await self._upstream_error_response(
                upstream_resp, trace=trace
            )

        if self.config.verbose:
            LOG.info(
                "upstream response status=%s stream=%s elapsed_ms=%s",
                upstream_status,
                bool(prepared.payload.get("stream")),
                elapsed_ms(started),
            )

        try:
            if prepared.payload.get("stream"):
                response = await self._proxy_streaming_response(
                    upstream_resp,
                    request,
                    prepared.original_model,
                    prepared.payload["messages"],
                    prepared.cache_namespace,
                    prepared.recovery_notice,
                    trace=trace,
                    record_response_scope=prepared.record_response_scope,
                    record_response_messages=prepared.record_response_messages,
                    record_response_contexts=prepared.record_response_contexts,
                )
            else:
                response = await self._proxy_regular_response(
                    upstream_resp,
                    prepared.original_model,
                    prepared.payload["messages"],
                    prepared.cache_namespace,
                    prepared.recovery_notice,
                    trace=trace,
                    record_response_scope=prepared.record_response_scope,
                    record_response_messages=prepared.record_response_messages,
                    record_response_contexts=prepared.record_response_contexts,
                )
            spinner.stop()
            log_stats_summary(
                getattr(response, "_usage", None)
            )
            self._finish_trace(
                trace,
                "completed",
                http_status=upstream_status,
                stream=bool(prepared.payload.get("stream")),
            )
            return response
        except (ConnectionResetError, ConnectionAbortedError):
            spinner.stop()
            LOG.info(
                "client disconnected during streaming path=%s", request_path
            )
            self._finish_trace(
                trace,
                "client_disconnected",
                http_status=upstream_status,
                stream=bool(prepared.payload.get("stream")),
            )
            # Return a minimal response (the client is already gone)
            return web.json_response(
                {"error": {"message": "Client disconnected"}}, status=499
            )
        finally:
            spinner.stop()

    # ── Catch-all for unsupported POST paths ───────────────────────────

    async def unsupported_post_path(
        self, request: web.Request
    ) -> web.StreamResponse | web.Response:
        """Handle POST requests to paths other than /chat/completions.
        aiohttp's router would otherwise return 404 before reaching this
        handler, preventing trace writers from capturing the request."""
        trace = self._start_trace(request.path, request)
        await self._record_request_body_for_trace(request, trace)
        self._finish_trace(trace, "rejected", http_status=404)
        return web.json_response(
            {"error": {"message": "Only /v1/chat/completions is supported"}},
            status=404,
        )

    # ── Regular (non-streaming) response ───────────────────────────────

    async def _proxy_regular_response(
        self,
        upstream_resp: ClientResponse,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
        record_response_scope: str | None = None,
        record_response_messages: list[dict[str, Any]] | None = None,
        record_response_contexts: list[tuple[str, list[dict[str, Any]]]]
        | None = None,
    ) -> web.Response:
        body = await upstream_resp.read()
        upstream_body = body
        usage = usage_from_body(upstream_body)
        try:
            body = rewrite_response_body(
                body,
                original_model,
                self.reasoning_store,
                request_messages,
                cache_namespace,
                content_prefix=recovery_notice,
                scope=record_response_scope,
                prior_messages=record_response_messages,
                recording_contexts=record_response_contexts,
                display_reasoning=self.config.display_reasoning,
                collapsible_reasoning=self.config.collapsible_reasoning,
            )
        except (ValueError, UnicodeDecodeError) as exc:
            LOG.warning(
                "failed to rewrite upstream JSON response: %s", exc
            )

        if self.reasoning_store is not None:
            self.reasoning_store.flush()

        if self.config.verbose:
            log_bytes("cursor response body", body)

        content_type = upstream_resp.headers.get(
            "Content-Type", "application/json"
        )
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }

        if trace is not None:
            trace.record_upstream_response(
                status=upstream_resp.status,
                headers=response_headers(upstream_resp),
                body=upstream_body,
                stream=False,
            )
            try:
                upstream_payload = orjson.loads(upstream_body)
            except (ValueError, UnicodeDecodeError):
                upstream_payload = None
            if isinstance(upstream_payload, dict):
                trace.record_usage(upstream_payload.get("usage"))
            trace.record_cursor_response(
                status=upstream_resp.status,
                headers=headers,
                body=body,
            )

        resp = web.Response(
            status=upstream_resp.status,
            headers=headers,
            body=body,
        )
        resp._usage = usage  # type: ignore[attr-defined]
        return resp

    # ── Streaming response ─────────────────────────────────────────────

    async def _proxy_streaming_response(
        self,
        upstream_resp: ClientResponse,
        request: web.Request,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
        record_response_scope: str | None = None,
        record_response_messages: list[dict[str, Any]] | None = None,
        record_response_contexts: list[tuple[str, list[dict[str, Any]]]]
        | None = None,
    ) -> web.StreamResponse:
        if trace is not None:
            trace.record_upstream_response(
                status=upstream_resp.status,
                headers=response_headers(upstream_resp),
                stream=True,
            )

        response = web.StreamResponse(status=upstream_resp.status)
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "close"
        if self.config.cors:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = (
                "POST, GET, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Origin, Content-Type, Accept, Authorization"
            )
            response.headers["Access-Control-Expose-Headers"] = "Content-Length"
            response.headers["Access-Control-Allow-Credentials"] = "true"

        if trace is not None:
            trace.record_cursor_response(
                status=upstream_resp.status,
                headers={
                    str(k): str(v) for k, v in response.headers.items()
                },
            )

        await response.prepare(request)

        accumulator = StreamAccumulator()
        usage: dict[str, Any] | None = None
        display_adapter = (
            CursorReasoningDisplayAdapter(self.config.collapsible_reasoning)
            if self.config.display_reasoning
            else None
        )
        scope = (
            record_response_scope
            if record_response_scope is not None
            else conversation_scope(request_messages, cache_namespace)
        )
        response_prior_messages = (
            record_response_messages
            if record_response_messages is not None
            else request_messages
        )
        response_contexts = (
            record_response_contexts
            if record_response_contexts is not None
            else [(scope, response_prior_messages)]
        )
        finalized = False
        pending_recovery_notice = recovery_notice
        try:
            while True:
                try:
                    line = await upstream_resp.content.readline()
                except (OSError, asyncio.TimeoutError) as exc:
                    LOG.warning(
                        "upstream streaming response read failed: %s", exc
                    )
                    break
                if not line:
                    break
                (
                    rewritten,
                    finalized,
                    pending_recovery_notice,
                    chunk_usage,
                ) = self._rewrite_sse_line(
                    line,
                    original_model,
                    accumulator,
                    cache_namespace,
                    response_contexts,
                    display_adapter,
                    pending_recovery_notice,
                    trace,
                )
                if chunk_usage is not None:
                    usage = chunk_usage
                if trace is not None:
                    trace.record_stream_chunk(line, rewritten)
                try:
                    await response.write(rewritten)
                except (ConnectionResetError, ConnectionAbortedError):
                    LOG.warning(
                        "client disconnected while sending streaming response chunk"
                    )
                    return response
                if finalized:
                    break
        finally:
            # Store partial reasoning when the stream exits without [DONE]
            if not finalized:
                if self.config.verbose:
                    log_json(
                        "model streaming assistant messages",
                        accumulator.messages(),
                    )
                stored = sum(
                    accumulator.store_reasoning(
                        self.reasoning_store,
                        ctx_scope,
                        cache_namespace,
                        prior_messages,
                    )
                    for ctx_scope, prior_messages in response_contexts
                )
                if self.config.verbose and stored:
                    LOG.info(
                        "stored %s streaming reasoning cache key(s) before exit",
                        stored,
                    )
            if self.reasoning_store is not None:
                self.reasoning_store.flush()

        response._usage = usage  # type: ignore[attr-defined]
        return response

    # ── SSE processing (synchronous, pure logic only) ──────────────────

    def _rewrite_sse_line(
        self,
        line: bytes,
        original_model: str,
        accumulator: StreamAccumulator,
        cache_namespace: str,
        response_contexts: list[tuple[str, list[dict[str, Any]]]],
        display_adapter: CursorReasoningDisplayAdapter | None,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
    ) -> tuple[bytes, bool, str | None, dict[str, Any] | None]:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return line, False, recovery_notice, None

        data = stripped[len(b"data:") :].strip()
        if data == b"[DONE]":
            return self._handle_done(
                accumulator,
                cache_namespace,
                response_contexts,
                display_adapter,
                original_model,
                recovery_notice,
            )

        # Fast path: skip JSON parsing when no modification is needed
        if (
            not self.config.display_reasoning
            and self.reasoning_store is None
            and original_model == self.config.upstream_model
        ):
            return line, False, recovery_notice, None

        try:
            chunk = orjson.loads(data)
        except (ValueError, UnicodeDecodeError):
            return line, False, recovery_notice, None

        if isinstance(chunk, dict):
            if (
                recovery_notice
                and inject_recovery_notice(chunk, recovery_notice)
            ):
                recovery_notice = None
            accumulator.ingest_chunk(chunk)
            stored = sum(
                accumulator.store_ready_reasoning(
                    self.reasoning_store,
                    scope,
                    cache_namespace,
                    prior_messages,
                )
                for scope, prior_messages in response_contexts
            )
            if self.config.verbose and stored:
                LOG.info(
                    "stored %s streaming reasoning cache key(s)", stored
                )
            chunk_usage = chunk.get("usage")
            if trace is not None:
                trace.record_usage(chunk_usage)
            if display_adapter is not None:
                display_adapter.rewrite_chunk(chunk)
            if "model" in chunk:
                chunk["model"] = original_model
            ending = b"\r\n" if line.endswith(b"\r\n") else b"\n"
            return (
                b"data: " + orjson.dumps(chunk) + ending,
                False,
                recovery_notice,
                chunk_usage if isinstance(chunk_usage, dict) else None,
            )
        return line, False, recovery_notice, None

    def _handle_done(
        self,
        accumulator: StreamAccumulator,
        cache_namespace: str,
        response_contexts: list[tuple[str, list[dict[str, Any]]]],
        display_adapter: CursorReasoningDisplayAdapter | None,
        original_model: str,
        recovery_notice: str | None,
    ) -> tuple[bytes, bool, str | None, dict[str, Any] | None]:
        if self.config.verbose:
            log_json(
                "model streaming assistant messages", accumulator.messages()
            )
        stored = sum(
            accumulator.store_reasoning(
                self.reasoning_store,
                scope,
                cache_namespace,
                prior_messages,
            )
            for scope, prior_messages in response_contexts
        )
        if self.config.verbose and stored:
            LOG.info(
                "stored %s streaming reasoning cache key(s)", stored
            )
        prefix = b""
        if display_adapter is None:
            if recovery_notice:
                prefix += sse_data(
                    recovery_notice_chunk(original_model, recovery_notice)
                )
            return prefix + b"data: [DONE]\n\n", True, None, None
        closing_chunk = display_adapter.flush_chunk(original_model)
        if closing_chunk is not None:
            prefix += sse_data(closing_chunk)
        if recovery_notice:
            prefix += sse_data(
                recovery_notice_chunk(original_model, recovery_notice)
            )
        return prefix + b"data: [DONE]\n\n", True, None, None

    # ── Trace helpers ──────────────────────────────────────────────────

    def _start_trace(
        self, request_path: str, request: web.Request
    ) -> TraceRequest | None:
        writer = self.trace_writer
        if writer is None:
            return None
        try:
            return writer.start_request(
                method=request.method,
                path=request_path,
                client_address=request.remote or "",
                headers={
                    str(name): str(value)
                    for name, value in request.headers.items()
                },
            )
        except OSError as exc:
            LOG.warning("failed to start request trace: %s", exc)
            return None

    def _finish_trace(
        self,
        trace: TraceRequest | None,
        status: str,
        **extra: Any,
    ) -> None:
        if trace is None:
            return
        try:
            trace.finish(status, **extra)
        except Exception as exc:
            LOG.warning("failed to write request trace: %s", exc)

    # ── Upstream helpers ───────────────────────────────────────────────

    def _cursor_authorization(
        self, request: web.Request
    ) -> str | None:
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            return f"Bearer {deepseek_api_key.strip()}"
        auth_header = request.headers.get("Authorization", "")
        scheme, separator, token = auth_header.strip().partition(" ")
        if (
            separator != " "
            or scheme.lower() != "bearer"
            or not token.strip()
        ):
            return None
        return f"Bearer {token.strip()}"

    def _upstream_headers(
        self, stream: bool, authorization: str
    ) -> dict[str, str]:
        return {
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": (
                "text/event-stream" if stream else "application/json"
            ),
            "Accept-Encoding": "identity",
            "User-Agent": SERVER_VERSION,
        }

    async def _upstream_error_response(
        self,
        upstream_resp: ClientResponse,
        *,
        trace: TraceRequest | None = None,
    ) -> web.Response:
        body = await upstream_resp.read()
        # Always log a summary of 4xx errors (client errors are often
        # actionable and the body contains the upstream error message).
        try:
            error_obj = orjson.loads(body)
            error_msg = (
                error_obj.get("error", {}).get("message", "")
                or error_obj.get("error", "")
            )
            if error_msg:
                LOG.warning(
                    "upstream error body: %s",
                    str(error_msg)[:300],
                )
        except Exception:
            log_bytes("upstream error body", body)
        if self.config.verbose:
            log_bytes("upstream error body", body)
        content_type = upstream_resp.headers.get(
            "Content-Type", "application/json"
        )
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }
        if trace is not None:
            trace.record_upstream_response(
                status=upstream_resp.status,
                headers=response_headers(upstream_resp),
                body=body,
            )
            trace.record_cursor_response(
                status=upstream_resp.status, headers=headers, body=body
            )
        return web.Response(
            status=upstream_resp.status, headers=headers, body=body
        )

    async def _read_json_body(
        self, request: web.Request
    ) -> dict[str, Any]:
        try:
            length = int(request.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0:
            raise ValueError("Invalid Content-Length")
        if length > self.config.max_request_body_bytes:
            raise RequestBodyTooLarge(
                f"Request body is too large; limit is "
                f"{self.config.max_request_body_bytes} bytes"
            )
        raw_body = await request.read()
        if not raw_body:
            raise ValueError("Request body is empty")
        try:
            payload = orjson.loads(raw_body)
        except ValueError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    async def _record_request_body_for_trace(
        self, request: web.Request, trace: TraceRequest | None
    ) -> None:
        if trace is None:
            return
        try:
            length = int(request.headers.get("Content-Length") or 0)
        except ValueError:
            trace.record_cursor_body_omitted(
                reason="invalid_content_length"
            )
            return
        if length < 0:
            trace.record_cursor_body_omitted(
                reason="invalid_content_length", body_bytes=length
            )
            return
        if length > self.config.max_request_body_bytes:
            trace.record_cursor_body_omitted(
                reason="body_too_large", body_bytes=length
            )
            return
        try:
            raw_body = await request.read()
        except OSError as exc:
            trace.record_cursor_body_omitted(
                reason=f"read_failed:{exc}", body_bytes=length
            )
            return
        trace.record_cursor_body_bytes(raw_body)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    config: ProxyConfig,
    reasoning_store: ReasoningStore,
    trace_writer: TraceWriter | None = None,
) -> web.Application:
    """Build and return a configured aiohttp ``Application``."""
    handler = DeepSeekProxyHandler(config, reasoning_store, trace_writer)
    # aiohttp defaults client_max_size to 1 MiB; align with our config (20 MiB).
    app = web.Application(client_max_size=config.max_request_body_bytes)
    app._handler = handler

    async def on_startup(app: web.Application) -> None:
        await handler.start()

    async def on_shutdown(app: web.Application) -> None:
        await handler.stop()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/healthz", handler.healthz)
    app.router.add_get("/v1/healthz", handler.healthz)
    app.router.add_get("/models", handler.models)
    app.router.add_get("/v1/models", handler.models)
    app.router.add_post("/chat/completions", handler.chat_completions)
    app.router.add_post(
        "/v1/chat/completions", handler.chat_completions
    )
    # Catch-all POST for unsupported paths — ensures traces are written
    # instead of being swallowed by aiohttp's built-in 404 handler.
    app.router.add_post(
        "/{tail:.*}", handler.unsupported_post_path
    )

    return app


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local DeepSeek Cursor proxy"
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help=f"YAML config file, default {default_config_path()}",
    )
    parser.add_argument(
        "--host", help="Bind host, default from config or 127.0.0.1"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Bind port, default from config or 9000",
    )
    parser.add_argument(
        "--model",
        help=(
            "Fallback DeepSeek model when the request has no model, "
            "default from config or deepseek-v4-pro"
        ),
    )
    parser.add_argument(
        "--base-url",
        help=(
            "DeepSeek base URL, "
            "default from config or https://api.deepseek.com"
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        help="DeepSeek thinking mode, default from config or enabled",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "max", "xhigh"],
        help="DeepSeek reasoning effort, default from config or max",
    )
    parser.add_argument(
        "--reasoning-content-path",
        type=Path,
        help=(
            "SQLite reasoning_content cache path, "
            f"default {default_reasoning_content_path()}"
        ),
    )
    parser.add_argument(
        "--ngrok",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start an ngrok tunnel and print the Cursor base URL",
    )
    parser.add_argument(
        "--ngrok-url",
        metavar="URL",
        help=(
            "Pass --url=URL to ngrok (reserved endpoint / custom domain); "
            "see `ngrok http --help`"
        ),
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log detailed request metadata and full payloads",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Write full structured request traces to this directory",
    )
    parser.add_argument(
        "--display-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mirror reasoning_content into Cursor-visible content",
    )
    parser.add_argument(
        "--collapsible-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use Markdown details for mirrored reasoning "
            "when display is enabled"
        ),
    )
    parser.add_argument(
        "--collasible-reasoning",
        "--collasible-resoning",
        dest="collapsible_reasoning",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-collasible-reasoning",
        "--no-collasible-resoning",
        dest="collapsible_reasoning",
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send permissive CORS headers",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        help="Upstream request timeout in seconds, default from config or 300",
    )
    parser.add_argument(
        "--max-request-body-bytes",
        type=int,
        help="Maximum accepted request body size, default from config",
    )
    parser.add_argument(
        "--reasoning-cache-max-age-seconds",
        type=int,
        help="Maximum reasoning cache row age in seconds, default from config",
    )
    parser.add_argument(
        "--reasoning-cache-max-rows",
        type=int,
        help="Maximum reasoning cache rows, default from config",
    )
    parser.add_argument(
        "--missing-reasoning-strategy",
        choices=["recover", "reject"],
        help=(
            "What to do when required reasoning_content is missing: "
            "recover (friendly default) or reject (strict debugging mode)"
        ),
    )
    parser.add_argument(
        "--clear-reasoning-cache",
        action="store_true",
        help="Clear the local reasoning_content SQLite cache and exit",
    )
    return parser


# ---------------------------------------------------------------------------
# Pure helpers (sync, no I/O)
# ---------------------------------------------------------------------------


def elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def log_json(label: str, payload: Any) -> None:
    LOG.info(
        "%s:\n%s",
        label,
        orjson.dumps(
            payload,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        ).decode("utf-8"),
    )


def log_bytes(label: str, body: bytes) -> None:
    try:
        payload = orjson.loads(body)
    except (ValueError, UnicodeDecodeError):
        LOG.info(
            "%s:\n%s", label, body.decode("utf-8", errors="replace")
        )
        return
    log_json(label, payload)


def usage_from_body(body: bytes) -> dict[str, Any] | None:
    try:
        payload = orjson.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            return usage
    return None


def log_cursor_request(
    payload: dict[str, Any],
    config: ProxyConfig,
) -> None:
    model = str(payload.get("model") or config.upstream_model)
    LOG.info(
        "\u250c request model=%s effort=%s messages=%s",
        model,
        config.reasoning_effort,
        format_count(message_count(payload)),
    )


def log_context_summary(prepared: Any) -> None:
    status = context_status(prepared)
    if status == "ok":
        LOG.info(
            "\u251c context status=ok reasoning_context=%s",
            format_count(prepared.patched_reasoning_messages),
        )
        return
    LOG.info(
        "\u251c context status=%s missing=%s recovered=%s dropped=%s",
        status,
        format_count(prepared.missing_reasoning_messages),
        format_count(prepared.recovered_reasoning_messages),
        format_count(prepared.recovery_dropped_messages),
    )


def log_send_summary(prepared: Any) -> None:
    LOG.info(
        "\u251c send    user_msgs=%s messages=%s tools=%s "
        "reasoning_content=%s",
        format_count(user_message_count(prepared.payload)),
        format_count(message_count(prepared.payload)),
        format_count(tool_count(prepared.payload)),
        format_count(reasoning_content_count(prepared.payload)),
    )


def log_stats_summary(usage: dict[str, Any] | None) -> None:
    LOG.info(
        "\u2514 stats   prompt=%s output=%s reasoning=%s cache_hit=%s",
        format_usage_count(usage, "prompt_tokens"),
        format_usage_count(usage, "completion_tokens"),
        format_count(reasoning_token_count(usage)),
        cache_hit_rate(usage),
    )


def context_status(prepared: Any) -> str:
    if prepared.recovered_reasoning_messages:
        return "recovered"
    if prepared.missing_reasoning_messages:
        return "missing"
    return "ok"


def message_count(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def tool_count(payload: dict[str, Any]) -> int:
    tools = payload.get("tools")
    return len(tools) if isinstance(tools, list) else 0


def user_message_count(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )


def reasoning_content_count(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("reasoning_content"), str)
    )


def format_usage_count(
    usage: dict[str, Any] | None, key: str
) -> str:
    if not isinstance(usage, dict):
        return "?"
    return format_count(usage.get(key))


def reasoning_token_count(usage: dict[str, Any] | None) -> Any:
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    return details.get("reasoning_tokens")


def cache_hit_rate(usage: dict[str, Any] | None) -> str:
    if not isinstance(usage, dict):
        return "?"
    hit_tokens = usage.get("prompt_cache_hit_tokens")
    miss_tokens = usage.get("prompt_cache_miss_tokens")
    if hit_tokens is None and miss_tokens is None:
        return "?"
    hit = int_or_zero(hit_tokens)
    miss = int_or_zero(miss_tokens)
    total = hit + miss
    if not total:
        return "?"
    return f"{hit / total:.1%}"


def format_count(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def sse_data(payload: dict[str, Any]) -> bytes:
    return b"data: " + orjson.dumps(payload) + b"\n\n"


def inject_recovery_notice(
    chunk: dict[str, Any], notice: str
) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if "content" not in delta and not delta.get("tool_calls"):
            continue
        existing_content = delta.get("content")
        delta["content"] = notice + (
            existing_content if isinstance(existing_content, str) else ""
        )
        return True
    return False


def recovery_notice_chunk(
    model: str,
    notice: str = RECOVERY_NOTICE_CONTENT,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-deepseek-cursor-proxy-recovery",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": notice},
                "finish_reason": None,
            }
        ],
    }


def summarize_chat_payload(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    tools = payload.get("tools")
    functions = payload.get("functions")
    return (
        f"model={payload.get('model')!r} "
        f"stream={bool(payload.get('stream'))} "
        f"messages={len(messages) if isinstance(messages, list) else 0} "
        f"tools={len(tools) if isinstance(tools, list) else 0} "
        f"functions={len(functions) if isinstance(functions, list) else 0} "
        f"tool_choice={payload.get('tool_choice')!r}"
    )


def read_response_body(
    response: Any,
    encoding: str | None = None,
) -> bytes:
    """Read body from a response-like object and decompress if needed.

    ``encoding`` can be passed explicitly (e.g. from aiohttp response
    headers).  If omitted the function falls back to
    ``response.headers["Content-Encoding"]`` (urllib style).
    """
    body = response.read()
    if encoding is None:
        headers = getattr(response, "headers", {})
        if hasattr(headers, "get"):
            encoding = (headers.get("Content-Encoding") or "").lower()
        else:
            encoding = ""
    enc = (encoding or "").lower()
    if enc == "gzip":
        return gzip.decompress(body)
    if enc == "deflate":
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -zlib.MAX_WBITS)
    return body


def response_headers(response: Any) -> dict[str, str]:
    """Extract headers dict from a response-like object."""
    headers = getattr(response, "headers", {})
    if hasattr(headers, "items"):
        return {str(name): str(value) for name, value in headers.items()}
    return {}


def warn_if_insecure_upstream(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return
    host = parsed.hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        return
    LOG.warning(
        "upstream base_url uses plain HTTP; bearer tokens may be exposed"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_config_from_args(
    args: argparse.Namespace,
) -> tuple[ProxyConfig, ReasoningStore, TraceWriter | None]:
    """Parse CLI args and wire up config, reasoning store and trace writer."""
    try:
        config = ProxyConfig.from_file(config_path=args.config_path)
    except ValueError as exc:
        configure_logging(verbose=bool(args.verbose))
        LOG.error("%s", exc)
        raise SystemExit(2) from exc

    updates: dict[str, Any] = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if args.model is not None:
        updates["upstream_model"] = args.model
    if args.base_url is not None:
        updates["upstream_base_url"] = args.base_url.rstrip("/")
    if args.thinking is not None:
        updates["thinking"] = args.thinking
    if args.reasoning_effort is not None:
        updates["reasoning_effort"] = args.reasoning_effort
    if args.reasoning_content_path is not None:
        updates["reasoning_content_path"] = args.reasoning_content_path
    if args.ngrok is not None:
        updates["ngrok"] = args.ngrok
    if args.ngrok_url is not None:
        stripped = str(args.ngrok_url).strip()
        updates["ngrok_url"] = stripped if stripped else None
    if args.verbose is not None:
        updates["verbose"] = args.verbose
    if args.trace_dir is not None:
        updates["trace_dir"] = args.trace_dir
    if args.display_reasoning is not None:
        updates["display_reasoning"] = args.display_reasoning
    if args.collapsible_reasoning is not None:
        updates["collapsible_reasoning"] = args.collapsible_reasoning
    if args.cors is not None:
        updates["cors"] = args.cors
    if args.request_timeout is not None:
        updates["request_timeout"] = args.request_timeout
    if args.max_request_body_bytes is not None:
        updates["max_request_body_bytes"] = args.max_request_body_bytes
    if args.reasoning_cache_max_age_seconds is not None:
        updates["reasoning_cache_max_age_seconds"] = (
            args.reasoning_cache_max_age_seconds
        )
    if args.reasoning_cache_max_rows is not None:
        updates["reasoning_cache_max_rows"] = args.reasoning_cache_max_rows
    if args.missing_reasoning_strategy is not None:
        updates["missing_reasoning_strategy"] = (
            args.missing_reasoning_strategy
        )
    if updates:
        config = replace(config, **updates)

    configure_logging(verbose=config.verbose)
    warn_if_insecure_upstream(config.upstream_base_url)

    store = ReasoningStore(
        config.reasoning_content_path,
        max_age_seconds=config.reasoning_cache_max_age_seconds,
        max_rows=config.reasoning_cache_max_rows,
    )

    trace_writer: TraceWriter | None = None
    if config.trace_dir is not None:
        try:
            trace_writer = TraceWriter(config.trace_dir)
        except OSError as exc:
            LOG.error("failed to initialize trace directory: %s", exc)
            store.close()
            raise SystemExit(2) from exc

    return config, store, trace_writer


async def async_main(args: argparse.Namespace) -> int:
    """Async entry point: start the aiohttp server and run until interrupt."""
    config, store, trace_writer = build_config_from_args(args)

    if args.clear_reasoning_cache:
        deleted = store.clear()
        LOG.info("cleared %s reasoning cache row(s)", deleted)
        store.close()
        return 0

    app = create_app(config, store, trace_writer)

    tunnel: NgrokTunnel | None = None
    public_url: str | None = None
    if config.ngrok:
        target_url = local_tunnel_target(config.host, config.port)
        tunnel = NgrokTunnel(target_url, ngrok_url=config.ngrok_url)
        try:
            public_url = tunnel.start()
        except RuntimeError as exc:
            LOG.error("%s", exc)
            store.close()
            return 2

    local_base_url = f"http://{config.host}:{config.port}/v1"
    api_base_url = (
        f"{public_url.rstrip('/')}/v1"
        if public_url is not None
        else local_base_url
    )

    LOG.info(
        "default_model: %s (%s, %s)",
        config.upstream_model,
        "thinking" if config.thinking == "enabled" else "no thinking",
        config.reasoning_effort,
    )

    if config.verbose:
        display_reasoning = "off"
        if config.display_reasoning:
            display_reasoning = (
                "on (collapsible)"
                if config.collapsible_reasoning
                else "on"
            )
        LOG.info("display_reasoning: %s", display_reasoning)
        LOG.info(
            "missing_reasoning_strategy: %s",
            config.missing_reasoning_strategy,
        )
        LOG.info("reasoning_cache: %s", config.reasoning_content_path)
        LOG.warning(
            "verbose logging enabled; "
            "prompts and code may be written to stdout"
        )
    if trace_writer is not None:
        LOG.info("trace_dir: %s", trace_writer.session_dir)
        LOG.warning(
            "trace logging enabled; "
            "prompts and code will be written to disk"
        )
    if public_url is None and not config.ngrok:
        LOG.info("public_tunnel: off")
    if config.verbose:
        LOG.info(
            "upstream_url: %s/chat/completions",
            config.upstream_base_url,
        )
    LOG.info("local_base_url: %s", local_base_url)
    LOG.info("api_base_url: %s", api_base_url)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.host, config.port)
    try:
        await site.start()
        LOG.info(
            "listening on http://%s:%s", config.host, config.port
        )
        # Sleep forever until interrupted
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        LOG.info("shutting down")
    finally:
        await runner.cleanup()
        if tunnel is not None:
            tunnel.stop()
        store.close()

    return 0


def main(argv: list[str] | None = None) -> int:
    """Synchronous entry point (wraps async_main)."""
    args = build_arg_parser().parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
