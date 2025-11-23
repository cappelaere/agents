"""
Langfuse tracing utilities for the Arctic Map Agent.

Author: Patrice G. Cappelaere, IBM Federal

These helpers centralize initialization and usage of the Langfuse SDK so that
FastAPI routes can record rich traces without embedding Langfuse-specific
logic everywhere.
"""

from langfuse import observe, Langfuse, get_client
import os, logging

logger = logging.getLogger("langfuse_utils")
logger.setLevel(logging.DEBUG)

secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
base_url = os.getenv("LANGFUSE_BASE_URL", "http://150.240.3.116:3000")

if not secret_key or not public_key:
    raise SystemExit("LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY not set")

langfuse = Langfuse(secret_key=secret_key, public_key=public_key, host=base_url)


def get_session_id(request):
    """Extract a best-effort session id from common HTTP header variants."""

    session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("session-id")
        or request.headers.get("session_id")
        or request.headers.get("x-sessionid")
        or request.headers.get("sessionid")
    )
    return session_id


def trace_start(request):
    """Start a Langfuse span for an incoming FastAPI request."""

    session_id = get_session_id(request)
    query_inputs = dict(request.query_params)
    inputs = {
        "session_id": session_id,
        "query": query_inputs,
        "method": request.method,
        "path": request.url.path,
    }
    trace = langfuse.start_span(name=request.url.path)
    trace.update(input=inputs)
    logger.info(inputs)
    return trace


def trace_end(trace, resp):
    """Finalize a Langfuse span with the given response payload."""

    try:
        trace.update(output=resp)
        trace.end()
    except Exception:
        logger.exception("langfuse flush failed")


def trace_flush():
    """Flush all pending Langfuse events to the backend."""

    langfuse.flush()