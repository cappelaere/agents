"""MCP bridge app that re-uses the AIS agent implementation.

This module makes the `agents/ais_agent` package importable at runtime,
imports its FastAPI `app` and mounts it under `/mcp` so the same routes and
OpenAPI are available at `/mcp/...`. It also exposes a lightweight
`/health` endpoint and logs clear errors if the import fails.
"""
import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger("mcp_server")
logging.basicConfig(level=logging.INFO)

# locate repo root relative to this file (may not be present inside container)
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Make repo root importable so `mcp_server.ais_agent` can be found when running inside container
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

# Bridge FastAPI app: docs and openapi are served under /mcp/*
app = FastAPI(
    title="MCP Bridge",
    version="0.1",
    docs_url="/mcp/docs",
    redoc_url="/mcp/redoc",
    openapi_url="/mcp/openapi.json",
)


# Attempt to import ais_agent robustly. In many container setups the
# repository path won't exist; prefer importing the package directly.
ais_agent = None
import_error = None

try:
    import ais_agent  # type: ignore
    logger.info("Imported ais_agent package directly")
    ais_agent = ais_agent
except Exception as e:
    import_error = e
    logger.info("Direct import of ais_agent failed: %s", e)

    # 1) Try env var 'AIS_AGENT_PATH' which can point to a mounted source dir
    env_path = os.getenv("AIS_AGENT_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            try:
                import ais_agent  # type: ignore
                logger.info("Imported ais_agent after adding AIS_AGENT_PATH: %s", sp)
                ais_agent = ais_agent
                import_error = None
            except Exception as e2:
                import_error = e2
                logger.exception("Import still failed after AIS_AGENT_PATH: %s", e2)
        else:
            logger.error("AIS_AGENT_PATH set but does not exist: %s", env_path)

    # 2) As a last resort: only attempt repo-relative path if present in container
    if ais_agent is None:
        default_agent_dir = REPO_ROOT / "agents" / "ais_agent"
        if default_agent_dir.exists():
            sp = str(default_agent_dir)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            try:
                import ais_agent  # type: ignore
                logger.info("Imported ais_agent from repo-relative path: %s", sp)
                ais_agent = ais_agent
                import_error = None
            except Exception as e3:
                import_error = e3
                logger.exception("Import failed from repo-relative path: %s", e3)
        else:
            logger.warning("No AIS_AGENT_PATH set and repo-relative ais_agent not present in container; \n"
                           "set AIS_AGENT_PATH to a mounted path containing ais_agent or install ais_agent as a package.")

    # 3) Try the bundled package under this mcp_server directory (mcp_server.ais_agent)
    if ais_agent is None:
        try:
            import importlib
            module = importlib.import_module("mcp_server.ais_agent")
            logger.info("Imported bundled mcp_server.ais_agent package")
            # the package's __init__ exports `app`
            ais_agent = module
            import_error = None
        except Exception as e_mod:
            logger.info("Failed to import bundled mcp_server.ais_agent: %s", e_mod)
            import_error = import_error or e_mod


# If we have the module, include or mount its routes; otherwise expose an erroring health endpoint
if ais_agent is not None:
    if hasattr(ais_agent, "app"):
        logger.info("Including ais_agent routes under /mcp")
        try:
            app.include_router(ais_agent.app.router, prefix="/mcp")
        except Exception:
            logger.exception("Including ais_agent router failed, falling back to mount")
            app.mount("/mcp", ais_agent.app)
    else:
        logger.error("ais_agent module imported but has no 'app' attribute")

    # health delegates to ais_agent.health if present
    @app.get("/health")
    async def health():
        health_fn = getattr(ais_agent, "health", None)
        if callable(health_fn):
            try:
                return await health_fn(None)
            except Exception as e:
                logger.exception("ais_agent.health failed: %s", e)
        return {"status": "ok", "source": "mcp_bridge"}

    # Serve a local copy of the AIS OpenAPI YAML at /mcp/openapi.yaml (if present)
    OPENAPI_YAML = REPO_ROOT / "mcp_server" / "ais_openapi.yaml"

    @app.get("/mcp/openapi.yaml")
    async def openapi_yaml():
        if OPENAPI_YAML.exists():
            return FileResponse(str(OPENAPI_YAML), media_type="application/yaml")
        return JSONResponse({"error": "openapi.yaml not found"}, status_code=404)

else:
    # Import failed; expose helpful health endpoint explaining missing package/import error
    logger.error("ais_agent could not be imported: %s", import_error)

    @app.get("/health")
    async def health_miss():
        msg = {
            "status": "error",
            "message": "ais_agent package not importable in this container",
            "hint": "Set AIS_AGENT_PATH env var to a mounted path with ais_agent or install it into the image",
        }
        if import_error:
            msg["import_error"] = str(import_error)
        return JSONResponse(msg, status_code=500)

