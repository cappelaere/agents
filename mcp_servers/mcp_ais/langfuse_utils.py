"""
Langfuse tracing utilities for the AIS FastMCP server.

Author: Patrice G. Cappelaere, IBM Federal

This module centralizes all interaction with the Langfuse Python SDK so that:
- Each MCP tool can easily start/end spans with consistent naming and payloads.
- The current MCP HTTP session id (if any) is attached to trace inputs.
- Transport details stay decoupled from tool business logic.
"""

import logging
import os
from typing import Any, Dict, Optional

from fastmcp.server import http as fastmcp_http  # type: ignore
from langfuse import Langfuse

logger = logging.getLogger("langfuse_utils")
logger.setLevel(logging.DEBUG)

secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
base_url = os.getenv("LANGFUSE_BASE_URL", "http://150.240.3.116:3000")

if not secret_key or not public_key:
    raise SystemExit("LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY not set")

langfuse = Langfuse(secret_key=secret_key, public_key=public_key, host=base_url)


def _current_session_id() -> Optional[str]:
    """
    Return the current HTTP MCP session id, if available.

    For Streamable HTTP transport, FastMCP stores the Starlette `Request`
    in a ContextVar; we read the `mcp-session-id` header from there.
    For STDIO transport (or non-HTTP calls), this safely returns ``None``.
    """
    try:
        request = fastmcp_http._current_http_request.get()  # type: ignore[attr-defined]
    except Exception:
        return None
    if request is None:
        return None
    return request.headers.get("mcp-session-id")


def trace_start(name: str, input: Dict[str, Any]):
    """
    Start a Langfuse span for an MCP tool invocation.

    - Uses ``name`` as the span name (typically the MCP tool name).
    - Records the given ``input`` payload on the span.
    - If running over HTTP Streamable transport, adds ``mcp_session_id``
      from the current request headers.
    """
    # Attach MCP session id to the trace input if we're running over HTTP.
    session_id = _current_session_id()
    if session_id:
        input = dict(input)  # shallow copy to avoid mutating caller data
        input["mcp_session_id"] = session_id

    trace = langfuse.start_span(name)
    trace.update(input=input)
    logger.info(f"name {name} inputs: {input}")
    return trace


def trace_end(trace, resp: Any):
    """
    Finalize a Langfuse span with the given response payload.

    This updates the span's ``output`` and calls ``end()``; any Langfuse
    errors are logged but not raised so they don't break MCP tool calls.
    """
    try:
        trace.update(output=resp)
        trace.end()
    except Exception:
        logger.exception("langfuse flush failed")


def trace_flush():
    """
    Flush all pending Langfuse events to the backend.

    This should be called periodically (or at the end of short-lived requests)
    to ensure traces are delivered promptly.
    """
    langfuse.flush()


