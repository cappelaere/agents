"""
Basic smoke tests for the MCP server running in Docker.

Why:
- Verify that the bridge container is up, serving health, docs, and AIS routes.
- Catch simple regressions in routing or OpenAPI wiring without hitting real AIS backends.

How to run (from repo root, with Docker running):
- Ensure mcp_server is up: `docker compose up -d mcp_server`
- Optionally override base URL: `export MCP_BASE_URL=http://localhost:8200`
- Run: `pytest test/test_mcp_server.py`
"""

import os

import httpx
import pytest


BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8200")


@pytest.fixture(scope="session")
def client():
    """Shared HTTP client for all tests."""
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as c:
        yield c


def test_bridge_health_ok(client: httpx.Client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") in {"ok", "degraded"}


def test_mcp_docs_and_openapi(client: httpx.Client):
    # OpenAPI JSON should be served at /mcp/openapi.json
    resp = client.get("/mcp/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("info", {}).get("title") == "MCP Bridge"
    assert "/mcp/ais/vessels/nearby" in data.get("paths", {})


def test_ais_health_via_bridge(client: httpx.Client):
    resp = client.get("/mcp/ais/health")
    assert resp.status_code == 200
    data = resp.json()
    # We don't assert on upstream keys here; just that the shape looks reasonable
    assert "status" in data
    assert "meta" in data


def test_list_aois(client: httpx.Client):
    resp = client.get("/mcp/ais/aoi")
    # If AOI file is missing, this could be 200 with empty items or 500;
    # we at least assert that the bridge doesn't 404.
    assert resp.status_code in {200, 500}
    if resp.status_code == 200:
        data = resp.json()
        assert "items" in data
        assert "meta" in data


def test_vessels_nearby_validation_error(client: httpx.Client):
    # radius_nm must be > 0; use an invalid value to trigger 422/400
    resp = client.get(
        "/mcp/ais/vessels/nearby",
        params={"lat": 0, "lon": 0, "radius_nm": -1},
    )
    assert resp.status_code in {400, 422}


