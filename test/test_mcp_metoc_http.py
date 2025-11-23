import json
import os

import httpx
import pytest


BASE_URL = os.getenv("MCP_METOC_URL", "http://localhost:8201/mcp")


def _initialize_session() -> str:
    """Perform MCP initialize and return the mcp-session-id from headers."""
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest-http", "version": "0.1"},
        },
    }

    resp = httpx.post(
        BASE_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        timeout=10.0,
    )

    assert resp.status_code == 200, resp.text
    session_id = resp.headers.get("mcp-session-id")
    assert session_id, "mcp-session-id header missing on initialize response"
    return session_id


def _post_jsonrpc_sse(session_id: str, method: str, params: dict) -> dict:
    """Helper to POST a JSON-RPC request and parse the JSON from SSE data line."""
    payload = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": method,
        "params": params,
    }

    resp = httpx.post(
        BASE_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id,
        },
        timeout=10.0,
    )

    assert resp.status_code == 200, resp.text

    # SSE response: find the 'data: {...}' line and parse the JSON payload
    for line in resp.text.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])

    raise AssertionError(f"No SSE data line found in response: {resp.text!r}")


@pytest.mark.integration
def test_http_tools_list():
    """Call the running MCP server over HTTP and verify tools/list works."""
    session_id = _initialize_session()

    data = _post_jsonrpc_sse(session_id, "tools/list", {})

    assert data.get("jsonrpc") == "2.0"
    assert "error" not in data
    assert "result" in data
    tools = data["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "get_metoc_health" in tool_names


@pytest.mark.integration
def test_http_get_metoc_health():
    """Call get_metoc_health via HTTP JSON-RPC against the Dockerized server."""
    session_id = _initialize_session()

    data = _post_jsonrpc_sse(
        session_id,
        "tools/call",
        {"name": "get_metoc_health", "arguments": {}},
    )

    assert data.get("jsonrpc") == "2.0"
    assert "error" not in data
    assert "result" in data


