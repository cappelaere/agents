## 🧩 Step 1 – Create the Virtual Environment

From your project root (where your agent code lives):

Python version 3.9.23

```bash
python3 -m venv venv
```

This creates a new isolated environment in a folder named `venv/`.
You can name it anything (e.g. `.env`, `.venv`, `agent_env`), but `venv` is standard.

---

## 🧠 Step 2 – Activate the Environment

### On **macOS / Linux**

```bash
source venv/bin/activate
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

# Run agents
docker compose up --no-deps -d metoc_agent
docker compose up --no-deps -d seaice_agent
docker compose up --no-deps -d ais_agent
docker compose up --no-deps -d ports_agent
docker compose up --no-deps -d map_agent
docker compose up --no-deps -d mcp_server

# follow live logs for a single service
docker compose logs -f metoc_agent
docker compose logs -f seaice_agent
docker compose logs -f ais_agent
docker compose logs -f ports_agent
docker compose logs -f map_agent

# last 200 lines + timestamps
docker compose logs --tail=200 --timestamps metoc_agent

# since a specific time (RFC3339 or relative)
docker compose logs --since=30m metoc_agent

# Down agent
docker down metoc_agent

## 🔗 MCP Server (AIS Bridge)

The `mcp_server` service is a thin FastAPI bridge that exposes the AIS agent
under a `/mcp/ais/...` prefix. It runs in its own container and delegates
all logic to the bundled `mcp_server/ais_agent` package.

- **Port**: 8200 (host) → 8200 (container)
- **Key endpoints**:
  - Bridge health: `GET http://localhost:8200/health`
  - AIS health via bridge: `GET http://localhost:8200/mcp/ais/health`
  - OpenAPI JSON: `GET http://localhost:8200/mcp/openapi.json`

Run just the MCP server with Docker:

```bash
docker compose build mcp_server
docker compose up --no-deps -d mcp_server
```

Quick smoke checks:

```bash
curl http://localhost:8200/health
curl http://localhost:8200/mcp/ais/health
curl http://localhost:8200/mcp/openapi.json | jq '.info.title'
```

### 🧪 MCP server tests

There is a small pytest module at `test/test_mcp_server.py` that verifies
the bridge health, docs, and a few AIS routes.

From the project root, with Docker running:

```bash
export MCP_BASE_URL=http://localhost:8200   # optional, this is the default
pytest test/test_mcp_server.py
```
