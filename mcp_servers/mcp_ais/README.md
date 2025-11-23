## AIS MCP Server (FastMCP)

This folder now contains a FastMCP-based AIS MCP server.

- **AIS logic is provided by the bundled `ais_agent` package.**
- The FastMCP application in `fastmcp_app.py` exposes selected AIS operations as MCP tools.
- The Streamable HTTP MCP endpoint is served at `/mcp` on port `8200`.

### Run locally (virtualenv)

```bash
python -m venv venv
source venv/bin/activate
cd mcp_servers/mcp_ais
pip install -r requirements.txt
python -m fastmcp_app
```

### Run with Docker

From the project root:

```bash
docker compose up -d mcp_ais
```

The container exposes the MCP **Streamable HTTP** endpoint at:

```text
POST http://localhost:8200/mcp
```

### Available MCP tools

The FastMCP server currently exposes (see `fastmcp_app.py`):

- `get_ais_health` – basic health/metadata about the AIS backend.
- `list_ais_aois` – list all configured AOIs and their geometry/metadata.
- `get_ais_aoi` – fetch a single AOI by id, with bbox/hash metadata.
- `ais_vessels_in_aoi` – list vessels inside a named AOI or arbitrary bbox.
- `ais_vessels_nearby` – list vessels within a radius around a point or AOI centroid.
- `get_vessel_info` – fetch vessel details by `mmsi`, `imo`, or `shipname`.

More tools can be added to `fastmcp_app.py` as needed, wrapping additional AIS agent functions.

### Using with MCP Inspector

1. Start the AIS MCP server (locally or via Docker as above).
2. Create or update your `mcp.json` to include:

   ```json
   {
     "mcpServers": {
       "AIS MCP Server": {
         "type": "streamable-http",
         "url": "http://localhost:8200/mcp",
         "note": "AIS MCP tools via FastMCP"
       }
     }
   }
   ```

3. Run MCP Inspector (example):

   ```bash
   npx @modelcontextprotocol/inspector --host 127.0.0.1 --port 6274
   ```

4. In Inspector, select **Streamable HTTP**, point it at `http://localhost:8200/mcp`, and connect.
   You should see the tools above under the Tools view and can invoke them interactively.

### HTTP integration tests

There is an HTTP-level test that exercises the running AIS FastMCP server: `test/test_mcp_ais_http.py`.

- Ensure Docker is running and the service is up:

  ```bash
  docker compose up -d mcp_ais
  ```

- From the project root, run:

  ```bash
  pytest -m integration test/test_mcp_ais_http.py
  ```

The tests will:

- Perform the MCP `initialize` handshake over Streamable HTTP.
- Call `tools/list` to verify the server advertises tools.
- Attempt a `tools/call` to ensure JSON-RPC responses are well-formed.
