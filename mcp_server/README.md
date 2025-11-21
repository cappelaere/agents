# Minimal MCP Server

This folder contains a minimal Model Context Protocol (MCP) server scaffold
implemented with FastAPI. It exposes a small `/mcp/run` endpoint and a `/health`
endpoint for quick smoke tests.

Run locally (recommended using a virtualenv):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

Run with Docker:

```bash
docker build -t mcp-server:local .
docker run -p 8080:8080 mcp-server:local
```

Quick test:

```bash
curl -sS -X POST http://localhost:8080/mcp/run -H 'Content-Type: application/json' \
  -d '{"input": {"q": "athens"}, "metadata": {"client": "test"}}' | jq
```

Health:

```bash
curl http://localhost:8080/health
```

Extend `app.py` to implement actual MCP behaviour and integrate with your
project (authentication, long-running jobs, model invocation, hooks, etc.).
