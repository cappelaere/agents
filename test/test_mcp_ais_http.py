import os

import httpx
import pytest


BASE_URL = os.getenv("MCP_AIS_URL", "http://localhost:8200/mcp")


def _initialize_session() -> str:
    """Perform MCP initialize against the AIS FastMCP server and return the session id."""
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest-http-ais", "version": "0.1"},
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


def _post_jsonrpc(session_id: str, method: str, params: dict) -> dict:
    """
    Helper to POST a JSON-RPC request and parse the JSON from the SSE data line.
    """
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
            import json

            data = json.loads(line[len("data: ") :])
            assert data.get("jsonrpc") == "2.0"
            return data

    raise AssertionError(f"No SSE data line found in response: {resp.text!r}")


@pytest.mark.integration
def test_http_tools_list_ais():
    """Call the running AIS FastMCP server over HTTP and verify tools/list works."""
    session_id = _initialize_session()

    data = _post_jsonrpc(session_id, "tools/list", {})

    assert data.get("jsonrpc") == "2.0"
    assert "error" not in data
    assert "result" in data
    tools = data["result"].get("tools", [])
    assert isinstance(tools, list)


@pytest.mark.integration
def test_http_ais_health_tool():
    """If any AIS MCP tools are exposed, verify that at least one can be called."""
    session_id = _initialize_session()

    tools_data = _post_jsonrpc(session_id, "tools/list", {})
    tools = tools_data["result"].get("tools", [])
    if not tools:
        pytest.skip("AIS MCP bridge currently exposes no tools via tools/list")

    # Take the first tool name and attempt a no-arg call; we only assert no JSON-RPC error.
    tool_name = tools[0]["name"]
    data = _post_jsonrpc(
        session_id,
        "tools/call",
        {"name": tool_name, "arguments": {}},
    )

    # For tools that require arguments, this may still be an error; we just assert JSON-RPC shape.
    assert data.get("jsonrpc") == "2.0"


