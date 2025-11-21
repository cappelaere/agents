## AIS MCP Bridge Server

This folder contains a small FastAPI application that acts as a **bridge**
between MCP-style clients and the AIS agent.

- **AIS routes are provided by the `ais_agent` FastAPI app.**
- The bridge mounts / includes those routes under the **`/mcp`** prefix.
- A lightweight `/health` endpoint is exposed for smoke tests.
- A static copy of the AIS OpenAPI spec is served at `/mcp/openapi.yaml`.

By default (see `docker-compose.yml`), the container mounts the main
`agents/ais_agent` source tree and imports that package. If that import
fails, it falls back to the bundled copy under `mcp_server/ais_agent`.

### Run locally (virtualenv)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Run with Docker

```bash
docker build -t mcp-server:local .
docker run -p 8080:8080 mcp-server:local
```

### Endpoints

- **Bridge health**

  ```bash
  curl http://localhost:8080/health
  ```

- **AIS health via bridge** (when `ais_agent` imports correctly)

  ```bash
  curl http://localhost:8080/mcp/ais/health
  ```

- **AIS OpenAPI spec**

  ```bash
  curl http://localhost:8080/mcp/openapi.yaml
  ```

The canonical AIS API is still defined by the AIS agent itself
(`agents/ais_agent`) and its `ais_openapi.yaml` spec. This bridge only
mounts that app under `/mcp` and makes it easy to consume from MCP clients.
