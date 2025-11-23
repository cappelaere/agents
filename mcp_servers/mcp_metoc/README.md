## Open-METOC MCP Server

`mcp_metoc` is a lightweight Model Context Protocol (MCP) server that wraps the
[Open‑Meteo](https://open-meteo.com) APIs (geocoding, atmosphere, marine, archive).
It provides a FastMCP server exposing Open-Meteo functionality as MCP tools.

### Features

- **MCP Tools** for Open-Meteo APIs:
  - Health check
  - Geocoding search
  - Atmospheric forecasts and historical data
  - Marine forecasts
- MCP endpoint served over Streamable HTTP at `/mcp`.

### Run locally

```bash
python -m venv venv
source venv/bin/activate
cd mcp_servers/mcp_metoc
pip install -r requirements.txt
python -m mcp_metoc.app
```

Environment overrides:

- `HOST` (default `0.0.0.0`)
- `PORT` (default `8201`)
- `MCP_TRANSPORT` (default `http`, as required by Streamable HTTP)

Once running, connect MCP Inspector to `http://localhost:8201/mcp` (Streamable HTTP).


### Docker

```bash
docker compose up -d mcp_metoc
```

Environment variables:

- `HOST` (default `0.0.0.0`)
- `PORT` (default `8201`)
- `LOG_LEVEL` (default `info`)
- `OPEN_METEO_TIMEOUT` (seconds, default `20`)
- `OPEN_METEO_USER_AGENT` (custom User‑Agent header for upstream calls)

### Testing

With the Docker service running (`docker compose up -d mcp_metoc`), you can run HTTP integration tests against the live MCP endpoint:

```bash
pytest -m integration test/test_mcp_metoc_http.py
```

By default the tests target `http://localhost:8201/mcp`. To point at a different host/port, set:

```bash
export MCP_METOC_URL="http://localhost:9000/mcp"
pytest -m integration test/test_mcp_metoc_http.py
```

These tests perform a full MCP Streamable HTTP handshake (`initialize` + `tools/list` / `tools/call`) over the running container.

### MCP tooling

The repository-level `mcp.json` contains a pre-configured entry:

```json
{
  "mcpServers": {
    "Open METOC Server": {
      "type": "streamable-http",
      "url": "http://localhost:8201/mcp",
      "note": "Open-Meteo tools via MCP"
    }
  }
}
```

Launch Inspector with:

```bash
npx @modelcontextprotocol/inspector --config /Users/patrice/Development/agents/mcp.json --host 127.0.0.1 --port 6274
```

