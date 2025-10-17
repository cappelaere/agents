
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Response, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Set
import json
import asyncio
import time

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

# --- Health endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok"}

# Optional legacy path if you ever used /healthz elsewhere
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def arctic_map(request: Request):
    """
    Default endpoint: renders a Leaflet map centered on the Arctic.
    """
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

@app.post("/ingest")
async def ingest_geojson(payload: Dict[str, Any] = Body(...)):
    """
    Receives a GeoJSON payload and broadcasts it to all connected map clients.
    Expected: FeatureCollection/Feature/Geometry object.
    """
    if not isinstance(payload, dict) or "type" not in payload:
        return Response(
            content=json.dumps({"error": "Invalid GeoJSON: missing 'type'"}),
            media_type="application/json",
            status_code=400
        )

    await manager.broadcast_text(json.dumps(payload))
    count = len(payload.get("features", [])) if isinstance(payload.get("features"), list) else None
    return {"status": "ok", "features": count}


# --- Version endpoint ---
import os

@app.get("/version")
async def version():
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
