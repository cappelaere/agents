# metoc_openmeteo_agent.py
# Minimal PoX API using Open-Meteo for atmosphere/marine and Open-Meteo Geocoding.
# Returns clean business JSON (no 'orchestrate' block). Each response includes a
# fully-qualified URL in 'endpoint' and governance.endpoint_url.

from fastapi import FastAPI, Request
from typing import Dict, Any, Optional
import os, uuid, socket, time, logging
from datetime import datetime
import httpx

app = FastAPI(title="Arctic METOC Agent API (PoX, Open‑Meteo + Geocoder)", version="0.4.2")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("metoc_openmeteo_agent")

APP_NAME = "metoc_agent"
APP_VERSION = "0.2.2"

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-MS"] = f"{dt_ms:.2f}"
    logger.info(f"{request.method} {request.url.path} {response.status_code} rid={rid} t_ms={dt_ms:.2f}")
    return response

def _gov(endpoint: str, req: Request, inputs: Dict[str, Any], lineage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "app": APP_NAME,
        "app_version": APP_VERSION,
        "endpoint": endpoint,
        "endpoint_url": str(req.url),  # fully-qualified URL
        "timestamp": _now_iso(),
        "request_id": getattr(req.state, "request_id", None) or req.headers.get("x-request-id"),
        "served_by": socket.gethostname(),
        "inputs": inputs,
        "data_lineage": lineage,
        "quality": {},
        "policies_applied": [],
        "risk_assessments": [],
        "tags": {"domain": "metoc", "provider": "open-meteo"},
        "license": {"name": "Open-Meteo", "url": "https://open-meteo.com/"},
    }

@app.get("/metoc/health", tags=["Health"])
async def health(request: Request):
    resp = {
        "status": "ok",
        "endpoint": str(request.url),  # fully-qualified URL in the top-level body
    }
    resp["governance"] = _gov("/metoc/health", request, {}, {})
    return resp

@app.get("/metoc/geocode/search", tags=["Geocoder"])
async def geocode_search(request: Request, name: str, count: int = 10, language: Optional[str] = None, format: str = "json"):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name, "count": count, "language": language, "format": format}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, params=params)
        data = r.json()
    resp = {
        "endpoint": str(request.url),
        "results": data,
    }
    resp["governance"] = _gov("/metoc/geocode/search", request, {"name": name, "count": count, "language": language, "format": format}, {"provider_url": url})
    return resp

@app.get("/metoc/atmosphere/forecast", tags=["Atmosphere"])
async def atmosphere_forecast(request: Request, lat: float, lon: float, hourly: Optional[str] = None, daily: Optional[str] = None,
                              current_weather: bool = True, timezone: Optional[str] = None, forecast_days: int = 7):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "hourly": hourly, "daily": daily, "current_weather": current_weather, "timezone": timezone, "forecast_days": forecast_days}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        data = r.json()
    resp = {
        "endpoint": str(request.url),
        "forecast": data,
    }
    resp["governance"] = _gov("/metoc/atmosphere/forecast", request, {"lat": lat, "lon": lon, "hourly": hourly, "daily": daily, "current_weather": current_weather, "timezone": timezone, "forecast_days": forecast_days}, {"provider_url": url})
    return resp

@app.get("/metoc/atmosphere/archive", tags=["Atmosphere"])
async def atmosphere_archive(request: Request, lat: float, lon: float, start_date: str, end_date: str, hourly: Optional[str] = None, daily: Optional[str] = None, timezone: Optional[str] = None):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date, "hourly": hourly, "daily": daily, "timezone": timezone}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        data = r.json()
    resp = {
        "endpoint": str(request.url),
        "archive": data,
    }
    resp["governance"] = _gov("/metoc/atmosphere/archive", request, {"lat": lat, "lon": lon, "start_date": start_date, "end_date": end_date, "hourly": hourly, "daily": daily, "timezone": timezone}, {"provider_url": url})
    return resp

@app.get("/metoc/marine/forecast", tags=["Marine"])
async def marine_forecast(request: Request, lat: float, lon: float, hourly: Optional[str] = None, timezone: Optional[str] = None, forecast_days: int = 5):
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {"latitude": lat, "longitude": lon, "hourly": hourly, "timezone": timezone, "forecast_days": forecast_days}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        data = r.json()
    resp = {
        "endpoint": str(request.url),
        "marine": data,
    }
    resp["governance"] = _gov("/metoc/marine/forecast", request, {"lat": lat, "lon": lon, "hourly": hourly, "timezone": timezone, "forecast_days": forecast_days}, {"provider_url": url})
    return resp
