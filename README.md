## 🧩 Step 1 – Create the Virtual Environment

From your project root (where your agent code lives):

Python version 3.9.23

I want 3.12
```
sudo dnf install -y python3.12 python3.12-devel python3.12-pip
```

```bash
python3.12 -m venv venv
```

This creates a new isolated environment in a folder named `venv/`.
You can name it anything (e.g. `.env`, `.venv`, `agent_env`), but `venv` is standard.

---

## 🧠 Step 2 – Activate the Environment

### On **macOS / Linux**

```bash
source venv/bin/activate
pip install --upgrade pip
```

Once activated, your shell prompt will change — e.g.

```
(venv) user@host:~/project$
```

All packages you install now go *inside* this environment, not system-wide.

---

## ⚙️ Step 3 – Upgrade pip (optional but recommended)

```bash
pip install --upgrade pip
```

---

## 📦 Step 4 – Install Required Libraries for Your Agents

For the **Open-Meteo + NSIDC + Governance PoX** agents, you’ll likely need:

```bash
pip install fastapi uvicorn httpx pydantic
pip install xarray netCDF4 pydap pyproj rioxarray numpy
pip install python-dotenv
```

If you’ll add LLM or Watsonx integration later:

```bash
pip install openai ibm-watsonx-ai
```

---

## 🧾 Step 5 – Freeze Requirements (for portability)

After verifying everything runs:

```bash
pip freeze > requirements.txt
```

This captures all exact library versions — useful for deployments or container builds.

---

## 🚀 Step 6 – Run Your Agentic API Locally

Example:

```bash
cd metoc_agent
uvicorn metoc_agent:app --port 8080 --reload

cd seaice_agent
uvicorn seaice_agent:app --port 8290 --reload

```

Then test:

```bash
curl http://localhost:8080/metoc/healthz
```

---

## 🧹 Step 7 – Deactivate When Done

```bash
deactivate
```

# macOS/Linux
Store Earhtdata login in .netrc
printf "machine urs.earthdata.nasa.gov login $USERNAME password $PASSWORD$\n" >> ~/.netrc
chmod 600 ~/.netrc

# ======================================================
# Virtual Machine
#
# Build procedure
docker compose build metoc_agent
docker compose build seaice_agent
docker compose build ais_agent
docker compose build ports_agent
docker compose build map_agent
docker compose build mcp_ais
docker compose build mcp_metoc

# Run agents
docker compose up --no-deps -d metoc_agent
docker compose up --no-deps -d seaice_agent
docker compose up --no-deps -d ais_agent
docker compose up --no-deps -d ports_agent
docker compose up --no-deps -d map_agent
docker compose up --no-deps -d mcp_ais
docker compose up --no-deps -d mcp_metoc

# follow live logs for a single service
docker compose logs -f metoc_agent
docker compose logs -f seaice_agent
docker compose logs -f ais_agent
docker compose logs -f ports_agent
docker compose logs -f map_agent
docker compose logs -f mcp_ais
docker compose logs -f mcp_metoc

# last 200 lines + timestamps
docker compose logs --tail=200 --timestamps metoc_agent

# since a specific time (RFC3339 or relative)
docker compose logs --since=30m metoc_agent

# Down agent
docker down metoc_agent
docker down mcp_ais
docker down mcp_metoc

## 🔗 MCP AIS Server (Bridge)

The `mcp_ais` service is a FastMCP-based MCP server that exposes the AIS agent
as MCP tools. It runs in its own container and delegates
all logic to the bundled `mcp_ais/ais_agent.py` module.

- **Port**: 8200 (host) → 8200 (container)
- **Key endpoints**:
  - Bridge health: `GET http://localhost:8200/health`
  - AIS health via bridge: `GET http://localhost:8200/mcp/ais/health`
  - OpenAPI JSON: `GET http://localhost:8200/mcp/openapi.json`

Run just the MCP AIS server with Docker:

```bash
docker compose build mcp_ais
docker compose up --no-deps -d mcp_ais
```

Quick smoke checks:

```bash
curl http://localhost:8200/health
curl http://localhost:8200/mcp/ais/health
curl http://localhost:8200/mcp/openapi.json | jq '.info.title'
```

## 🌤️ MCP METOC Server (Open‑Meteo)

`mcp_metoc` now uses the [`FastMCP`](https://gofastmcp.com) framework to expose
Open‑Meteo tools directly over Streamable HTTP (default port **8201**).

```bash
docker compose build mcp_metoc
docker compose up --no-deps -d mcp_metoc
```

Point MCP Inspector (Streamable HTTP) at `http://localhost:8201/mcp`.

Need plain REST + OpenAPI docs? Run the optional FastAPI app:

```bash
uvicorn mcp_metoc.api:app --host 0.0.0.0 --port 8301 --reload
curl 'http://localhost:8301/metoc/atmosphere/forecast?lat=71.29&lon=-156.79&hourly=temperature_2m'
```

### 🧪 MCP server tests

There is a small pytest module at `test/test_mcp_server.py` that verifies
the AIS bridge health, docs, and a few routes.

From the project root, with Docker running:

```bash
export MCP_BASE_URL=http://localhost:8200   # optional, this is the default
pytest test/test_mcp_server.py
```

There are also Streamable HTTP MCP integration tests similar to `mcp_metoc`:

```bash
export MCP_AIS_URL=http://localhost:8200/mcp   # optional, this is the default
pytest -m integration test/test_mcp_ais_http.py
```
