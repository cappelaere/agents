"""
Arctic Map Agent (Leaflet) - WebSocket-backed map overlay service.

Author: Patrice G. Cappelaere, IBM Federal

This FastAPI app serves a Leaflet-based Arctic map UI and exposes a simple
ingest endpoint and WebSocket channel so other agents can push GeoJSON
overlays to connected browsers in real time.
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Response, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Set
import json
import asyncio
import time
import os

from jsonschema import validate, ValidationError

app = FastAPI(title="Arctic Map Agent (Leaflet)")
templates = Jinja2Templates(directory="templates")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve static files (e.g., favicon)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.png")


# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory WebSocket connection registry
class ConnectionManager:
    """In-memory registry for active WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast_text(self, message: str):
        async with self._lock:
            dead = []
            for ws in list(self.active_connections):
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.active_connections.discard(ws)

manager = ConnectionManager()


# --- GeoJSON validation & LLM patching helpers ---

GEOJSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Basic GeoJSON FeatureCollection",
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["FeatureCollection", "Feature", "Point", "LineString", "Polygon"],
        },
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "geometry"],
                "properties": {
                    "type": {"type": "string", "const": "Feature"},
                    "properties": {"type": ["object", "null"]},
                    "geometry": {
                        "type": ["object", "null"],
                        "required": ["type", "coordinates"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["Point", "LineString", "Polygon"],
                            },
                            "coordinates": {},
                        },
                    },
                },
            },
        },
    },
}


def _patch_geojson_from_llm(payload: Any) -> Dict[str, Any]:
    """Attempt to coerce an LLM-produced structure into valid GeoJSON.

    This applies a series of safe, conservative fixes:
    - Unwraps common extra nesting keys such as ``{\"geojson\": {...}}``.
    - Normalizes ``type`` casing (e.g., ``featurecollection`` → ``FeatureCollection``).
    - Wraps bare Feature/Geometry objects into a ``FeatureCollection``.
    - Drops obviously invalid features.

    Args:
        payload: Raw JSON-like structure from the client/LLM.

    Returns:
        dict: A patched GeoJSON object suitable for schema validation.

    Raises:
        ValueError: If the payload cannot reasonably be interpreted as GeoJSON.
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    # Unwrap common LLM wrappers like {"geojson": {...}} or {"data": {...}}
    for key in ("geojson", "GeoJSON", "data"):
        inner = payload.get(key)
        if isinstance(inner, dict) and "type" in inner:
            payload = inner
            break

    # Normalize top-level type casing
    if "type" in payload and isinstance(payload["type"], str):
        t = payload["type"].strip()
        lowered = t.lower()
        if lowered == "featurecollection":
            payload["type"] = "FeatureCollection"
        elif lowered == "feature":
            payload["type"] = "Feature"

    # If this is a single Feature, wrap it in a FeatureCollection
    if payload.get("type") == "Feature":
        payload = {
            "type": "FeatureCollection",
            "features": [payload],
        }

    # If this looks like a bare geometry, wrap it as a FeatureCollection[Feature]
    if payload.get("type") in ("Point", "LineString", "Polygon"):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": payload,
                    "properties": {},
                }
            ],
        }

    # Ensure features is a list when we claim to be a FeatureCollection
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features")
        if features is None:
            payload["features"] = []
        elif not isinstance(features, list):
            payload["features"] = [features] if isinstance(features, dict) else []

        cleaned_features = []
        for feat in payload["features"]:
            if not isinstance(feat, dict):
                continue
            if feat.get("type") != "Feature":
                feat["type"] = "Feature"
            geom = feat.get("geometry")
            if geom is None or not isinstance(geom, dict):
                continue
            if "type" not in geom or "coordinates" not in geom:
                continue
            cleaned_features.append(feat)
        payload["features"] = cleaned_features

    return payload

# --- Health endpoints ---
@app.get("/health")
async def health():
    """Basic liveness probe for the map agent."""
    return {"status": "ok"}

# ---------- Home Page ---------------
@app.get("/", response_class=HTMLResponse)
async def arctic_map(request: Request):
    """Render the main Leaflet map page centered on the Arctic."""
    return templates.TemplateResponse("index.html", {"request": request})

async def _keepalive_task(websocket: WebSocket, interval_sec: int = 25):
    """Periodically send a ping-ish text frame so proxies don't kill idle sockets."""
    try:
        while True:
            await asyncio.sleep(interval_sec)
            payload = {"type": "ping", "ts": time.time()}
            await websocket.send_text(json.dumps(payload))
    except Exception:
        # Exit quietly on disconnect or send failure
        return

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint used by browser clients to receive map updates."""
    await manager.connect(websocket)
    keepalive = asyncio.create_task(_keepalive_task(websocket))
    try:
        while True:
            # We accept (and ignore) any incoming text to keep the connection active both ways.
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        keepalive.cancel()
        await manager.disconnect(websocket)

# -------------- Ingest ----------------------
@app.post("/ingest")
async def ingest_geojson(request: Request):
    """Broadcast a GeoJSON payload to all connected map clients.

    Expected: a GeoJSON FeatureCollection, Feature, or Geometry object.

    Since this endpoint is typically fed by an LLM, we apply both:
    - a patching pass to coerce common LLM mistakes into valid GeoJSON, and
    - a JSON Schema validation step to reject irreparably invalid inputs.

    We also log a compact summary of what we received and what we broadcast to
    make debugging LLM behavior and map overlays easier.
    """

    # 1) Inspect the raw body
    raw = await request.body()
    logger.info("Raw /ingest body (first 500 bytes): %r", raw[:500])
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Empty request body on /ingest",
        )

    # 2) Try to parse as JSON
    try:
        payload = json.loads(raw)
    except Exception as exc:
        logger.warning("Invalid JSON on /ingest: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON: {exc}",
        )
    
    # 3) (optional) log high-level structure
    if isinstance(payload, dict):
        logger.info(
            "Parsed /ingest JSON: type=%s keys=%s",
            payload.get("type"),
            list(payload.keys())[:10],
        )
    else:
        logger.info(
            "Parsed /ingest JSON of type %s", type(payload)
        )

    # 1) Manually parse JSON so FastAPI doesn’t 422 before we run
    # try:
    #     logger.info("Received /ingest request from %s", request.client.host if request.client else "unknown")
    #     payload = await request.json()
    # except Exception as exc:
    #     logger.warning("Invalid JSON on /ingest: %s", exc)
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"Invalid JSON: {exc}",
    #     )
    
    # # Log high-level shape of the incoming payload without dumping entire blobs.
    # if isinstance(payload, dict):
    #     logger.info(
    #         "Received /ingest payload: type=%s keys=%s",
    #         payload.get("type"),
    #         list(payload.keys())[:10],
    #     )
    # else:
    #     logger.info(
    #         "Received /ingest non-object payload of type %s",
    #         type(payload),
    #    )

    try:
        patched = _patch_geojson_from_llm(payload)
        validate(instance=patched, schema=GEOJSON_SCHEMA)
    except (ValueError, ValidationError) as exc:
        logger.warning("Invalid GeoJSON on /ingest: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid GeoJSON: {exc}",
        )

    await manager.broadcast_text(json.dumps(patched))
    count = (
        len(patched.get("features", []))
        if isinstance(patched.get("features"), list)
        else None
    )
    logger.info(
        "Broadcasting patched GeoJSON on /ingest: type=%s features=%s",
        patched.get("type"),
        count,
    )
    response = {"status": "ok", "features": count}
    return response


# --- Version endpoint ---
@app.get("/version")
async def version():
    """Return basic version/build information for the map agent."""
    return {
        "app": "Arctic Map Agent (Leaflet)",
        "version": os.getenv("APP_VERSION", "0.1.0"),
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "build_time": os.getenv("BUILD_TIME", "unknown")
    }

# --- Basic access logging middleware ---
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

logger = logging.getLogger("map_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        path = request.url.path
        method = request.method
        client = request.client.host if request.client else "-"
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            logger.exception(f"{client} {method} {path} -> 500 error: {e}")
            raise
        finally:
            elapsed_ms = int((time.time() - start) * 1000)
            logger.info(f"{client} {method} {path} -> {status} {elapsed_ms}ms")
        return response

app.add_middleware(AccessLogMiddleware)
