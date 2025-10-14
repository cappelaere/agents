
# seaice_agent.py
# NSIDC Sea Ice Agent (runtime fetch) for NRT CDR-style Arctic daily grids.
# Includes request-id middleware, governance metadata in responses, and optional
# watsonx.governance (OpenScale payload logging) integration via environment variables.

from fastapi import FastAPI, Query, HTTPException, Body, Request
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import os
import pathlib
import httpx
import uuid
import socket
import math
import time
import logging

import numpy as np
from pyproj import Transformer

app = FastAPI(title="NSIDC Sea Ice Agent API (Runtime Fetch)", version="0.4.0")

# -----------------------------
# App/version and configuration
# -----------------------------
APP_NAME = "seaice_agent"
APP_VERSION = "0.2"

NSIDC_VAR_NAME = os.getenv("NSIDC_VAR_NAME", "cdr_seaice_conc")
NSIDC_CRS = os.getenv("NSIDC_CRS", "EPSG:3411")  # Arctic Polar Stereographic
NSIDC_DATA_DIR = os.getenv("NSIDC_DATA_DIR", "./data")

# URL pattern must support {yyyy} {mm} {dd} {sensor}, e.g.:
# https://host/path/{yyyy}/{mm}/{dd}/{sensor}.nc
NSIDC_URL_PATTERN = os.getenv(
    "NSIDC_URL_PATTERN",
    "https://noaadata.apps.nsidc.org/NOAA/G10016_V3/CDR/north/daily/{yyyy}/sic_psn25_{yyyymmdd}_{sensor}_icdr_v03r00.nc"
)

# Optional public WMS endpoint to help users visualize (template only)
NSIDC_WMS_BASE = os.getenv("NSIDC_WMS_BASE", "https://example.org/wms")
NSIDC_WMS_LAYER = os.getenv("NSIDC_WMS_LAYER", "seaice_conc")

# Try candidate sensors in order
SENSOR_CANDIDATES: List[str] = [s.strip() for s in os.getenv(
    "NSIDC_SENSORS", "SSMI, SSMIS, AMSR2"
).split(",") if s.strip()]

# Ensure data directory exists
pathlib.Path(NSIDC_DATA_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Logger
# -----------------------------
logger = logging.getLogger("seaice_agent")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# -----------------------------
# Optional watsonx.governance (OpenScale) config
# -----------------------------
WXG_ENABLED = os.getenv("WATSONX_GOV_ENABLED", "false").lower() in ("1","true","yes","y")
WXG_URL = os.getenv("WATSONX_OPENSCALE_PAYLOAD_URL")  # e.g., https://openscale.region.cloud.ibm.com/aiopenscale/instances/<id>/v1/data_mart/scoring_payloads
WXG_API_KEY = os.getenv("WATSONX_API_KEY")            # IBM Cloud API key
WXG_BEARER = os.getenv("WATSONX_BEARER_TOKEN")        # Optional: if provided, used directly
WXG_TIMEOUT = float(os.getenv("WATSONX_TIMEOUT_SEC", "5.0"))
WXG_DEPLOYMENT = os.getenv("WATSONX_DEPLOYMENT_NAME", "seaice-agent")

# -----------------------------
# Utilities
# -----------------------------
def _yyyymmdd_parts(date_str: str) -> Tuple[str, str, str]:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{dt.year:04d}", f"{dt.month:02d}", f"{dt.day:02d}"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date '{date_str}': {e}")


def _candidate_urls(date_str: str) -> List[Tuple[str, str]]:
    yyyy, mm, dd = _yyyymmdd_parts(date_str)
    yyyymmdd = f"{yyyy}{mm}{dd}"
    date_iso = f"{yyyy}-{mm}-{dd}"

    pattern = NSIDC_URL_PATTERN
    urls: List[Tuple[str, str]] = []

    for sensor in SENSOR_CANDIDATES:
        sensor_clean = sensor.replace(" ", "")
        values = {
            "yyyy": yyyy,
            "mm": mm,
            "dd": dd,
            "yyyymmdd": yyyymmdd,
            "date": date_iso,
            "sensor": sensor_clean,
            "sensor_lower": sensor_clean.lower(),
        }
        try:
            url = pattern.format_map(values)
        except KeyError as ke:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"NSIDC_URL_PATTERN has unknown placeholder: {{{ke.args[0]}}}. "
                    f"Supported placeholders: {list(values.keys())}. "
                    f"Pattern was: {pattern}"
                ),
            )
        urls.append((url, sensor))
    return urls


def _candidate_local_paths(date_str: str) -> List[Tuple[str, str]]:
    yyyy, mm, dd = _yyyymmdd_parts(date_str)
    local_paths = []
    for sensor in SENSOR_CANDIDATES:
        fname = f"seaice_{yyyy}{mm}{dd}_{sensor}.nc"
        lp = os.path.join(NSIDC_DATA_DIR, yyyy, mm, fname)
        local_paths.append((lp, sensor))
    return local_paths


def _ensure_local_file(date_str: str, force: bool = False) -> Tuple[str, Optional[str], bool]:
    """
    Ensure a local file exists for the given date. Try each sensor in order.
    Returns (local_path, sensor_used, removed_flag_if_force_replaced)
    """
    removed = False
    # check existing cached first (unless force)
    for (lp, sensor) in _candidate_local_paths(date_str):
        if os.path.exists(lp) and not force:
            return lp, sensor, removed

    # If force, remove any stale files for that date before re-downloading
    if force:
        for (lp, _) in _candidate_local_paths(date_str):
            if os.path.exists(lp):
                try:
                    os.remove(lp)
                    removed = True
                except Exception:
                    pass

    # Try download in order
    for (url, sensor) in _candidate_urls(date_str):
        (lp, _) = _candidate_local_paths(date_str)[SENSOR_CANDIDATES.index(sensor)]
        pathlib.Path(os.path.dirname(lp)).mkdir(parents=True, exist_ok=True)
        try:
            with httpx.stream("GET", url, timeout=60.0) as r:
                if r.status_code == 200:
                    with open(lp, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                    if os.path.getsize(lp) > 0:
                        return lp, sensor, removed
                # try next candidate sensor
        except Exception:
            # try next sensor
            continue

    raise HTTPException(
        status_code=404,
        detail=f"Unable to fetch dataset for {date_str} from any candidate sensor."
    )


# -----------------------------
# Dataset cache / open
# -----------------------------
# NOTE: Avoid forward-ref to xr.Dataset to prevent "XR is not defined"
_ds_cache: Dict[str, Any] = {}

def _open_ds(local_path: str):
    import xarray as xr  # local import so module loads without xarray at build
    if local_path in _ds_cache:
        return _ds_cache[local_path]
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail=f"Local file not found: {local_path}")
    ds = xr.open_dataset(local_path, engine="netcdf4")
    _ds_cache[local_path] = ds
    return ds


# -----------------------------
# Reprojection helpers
# -----------------------------
# Lat/Lon (EPSG:4326) -> NSIDC_CRS (default EPSG:3411)
_transformer_cache: Dict[str, Transformer] = {}

def _get_transformer() -> Transformer:
    if NSIDC_CRS not in _transformer_cache:
        _transformer_cache[NSIDC_CRS] = Transformer.from_crs("EPSG:4326", NSIDC_CRS, always_xy=True)
    return _transformer_cache[NSIDC_CRS]

def _to_proj(lon: float, lat: float) -> Tuple[float, float]:
    transformer = _get_transformer()
    x, y = transformer.transform(lon, lat)
    return x, y


def _nearest_index(ds, x: float, y: float, lon: Optional[float] = None, lat: Optional[float] = None) -> Dict[str, int]:
    """
    Compute nearest indices in projected space; assumes dataset has x/y or similar coords.
    """
    # Try common coord names
    x_name = next((c for c in ["x", "projection_x_coordinate", "lon_east", "xc", "cols"] if c in ds.coords), None)
    y_name = next((c for c in ["y", "projection_y_coordinate", "lat_north", "yc", "rows"] if c in ds.coords), None)
    if x_name is None or y_name is None:
        # Fall back to dimensions named x/y
        x_name = x_name or "x"
        y_name = y_name or "y"
        if x_name not in ds.dims or y_name not in ds.dims:
            raise HTTPException(status_code=500, detail="Dataset missing recognizable x/y coordinates.")

    xs = ds.coords[x_name].values
    ys = ds.coords[y_name].values

    ix = int(np.argmin(np.abs(xs - x)))
    iy = int(np.argmin(np.abs(ys - y)))
    return {x_name: ix, y_name: iy}


# -----------------------------
# Request ID + Timing middleware
# -----------------------------
def _get_req_id(request: Request) -> Optional[str]:
    # prefer middleware-populated request.state.request_id
    return getattr(request.state, "request_id", None) or request.headers.get("x-request-id")

@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("x-request-id")
    if not rid:
        rid = str(uuid.uuid4())
    request.state.request_id = rid

    t0 = time.perf_counter()
    response = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-MS"] = f"{dt_ms:.2f}"
    logger.info(f"{request.method} {request.url.path} {response.status_code} rid={rid} t_ms={dt_ms:.2f}")
    return response

# -----------------------------
# Governance helpers (Watsonx.governance friendly)
# -----------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def _gov_base(endpoint: str,
              request_id: Optional[str],
              inputs: Dict[str, Any],
              lineage: Dict[str, Any],
              quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "app": APP_NAME,
        "app_version": APP_VERSION,
        "endpoint": endpoint,
        "timestamp": _now_iso(),
        "request_id": request_id or str(uuid.uuid4()),
        "served_by": socket.gethostname(),
        "inputs": inputs,
        "data_lineage": lineage,
        "quality": quality or {},
        # Hooks for governance/policy engines
        "policies_applied": [],
        "risk_assessments": [],
        "tags": {"domain": "sea-ice", "provider": "NSIDC"},
        "license": {
            "name": "NSIDC NRT CDR (G10016 v3) (or equivalent source per URL pattern)",
            "url": "https://nsidc.org/",
        },
    }

async def _wxg_log_payload(governance: Dict[str, Any], output: Dict[str, Any]) -> None:
    """
    Best-effort OpenScale payload logging. No-op unless WXG_ENABLED and WXG_URL set.
    Schema kept generic: inputs/outputs + context in 'governance' block.
    """
    if not WXG_ENABLED or not WXG_URL:
        return
    headers = {"Content-Type": "application/json"}
    if WXG_BEARER:
        headers["Authorization"] = f"Bearer {WXG_BEARER}"
    elif WXG_API_KEY:
        headers["Authorization"] = f"Bearer {WXG_API_KEY}"
    payload = {
        "deployment": WXG_DEPLOYMENT,
        "timestamp": governance.get("timestamp"),
        "request_id": governance.get("request_id"),
        "inputs": governance.get("inputs"),
        "outputs": output,
        "context": {
            "endpoint": governance.get("endpoint"),
            "data_lineage": governance.get("data_lineage"),
            "quality": governance.get("quality"),
            "tags": governance.get("tags"),
            "served_by": governance.get("served_by"),
            "app": governance.get("app"),
            "app_version": governance.get("app_version"),
        },
    }
    try:
        async with httpx.AsyncClient(timeout=WXG_TIMEOUT) as client:
            await client.post(WXG_URL, json=payload, headers=headers)
    except Exception as e:
        logger.debug(f"OpenScale log skipped or failed: {e}")

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/seaice/health")
async def healthz(request: Request):
    resp = {
        "status": "ok",
        "variable": NSIDC_VAR_NAME,
        "crs": NSIDC_CRS,
        "data_dir": os.path.abspath(NSIDC_DATA_DIR),
        "url_pattern": NSIDC_URL_PATTERN,
        "sensors_tried": SENSOR_CANDIDATES,
    }
    resp["governance"] = _gov_base(
        endpoint="/seaice/health",
        request_id=_get_req_id(request),
        inputs={},
        lineage={"dataset_url_pattern": NSIDC_URL_PATTERN, "crs": NSIDC_CRS, "variable": NSIDC_VAR_NAME},
    )
    # No payload logging for health
    return resp


@app.get("/seaice/wms")
async def wms_template(
    layer: Optional[str] = None,
    time: Optional[str] = None,
    bbox: Optional[str] = "60,-180,90,180",
    srs: str = "EPSG:4326",
    width: int = 1024,
    height: int = 512,
    request: Request = None,
):
    lyr = layer or NSIDC_WMS_LAYER
    resp = {
        "wms_url_template": NSIDC_WMS_BASE,
        "params_example": {
            "service": "WMS",
            "version": "1.1.0",
            "request": "GetMap",
            "layers": lyr,
            "srs": srs,
            "bbox": bbox,
            "width": width,
            "height": height,
            "format": "image/png",
            "transparent": "true",
            "time": time or "YYYY-MM-DD",
        },
        "note": "Use in a WMS-capable client; adjust bbox/size/time to your viewport and date.",
    }
    resp["governance"] = _gov_base(
        endpoint="/seaice/wms",
        request_id=_get_req_id(request) if request else None,
        inputs={"layer": layer, "time": time, "bbox": bbox, "srs": srs, "width": width, "height": height},
        lineage={"wms_base": NSIDC_WMS_BASE, "layer": lyr},
    )
    # no payload logging for template
    return resp


@app.get("/seaice/download")
async def download(
    time: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
    force: bool = Query(False, description="If true, redownload even if cached"),
    request: Request = None,
):
    # Return cached if present
    for local_path, sensor in _candidate_local_paths(time):
        if os.path.exists(local_path) and not force:
            url = next(u for u, s in _candidate_urls(time) if s == sensor)
            sz = os.path.getsize(local_path)
            resp = {
                "status": "cached",
                "time": time,
                "sensor": sensor,
                "url": url,
                "file": os.path.abspath(local_path),
                "size_bytes": sz,
            }
            resp["governance"] = _gov_base(
                endpoint="/seaice/download",
                request_id=_get_req_id(request) if request else None,
                inputs={"time": time, "force": force},
                lineage={"sensor": sensor, "source_url": url, "local_file": os.path.abspath(local_path)},
            )
            try:
                await _wxg_log_payload(resp["governance"], output={"status": resp["status"], "file": resp["file"]})
            except Exception:
                pass
            return resp

    # Otherwise download
    local_path, sensor_used, removed = _ensure_local_file(time, force=force)
    url_used = next((u for u, s in _candidate_urls(time) if s == sensor_used), "inferred")
    resp = {
        "status": "downloaded",
        "time": time,
        "sensor": sensor_used,
        "url": url_used,
        "file": os.path.abspath(local_path),
        "size_bytes": os.path.getsize(local_path),
        "cache_cleared": removed,
    }
    resp["governance"] = _gov_base(
        endpoint="/seaice/download",
        request_id=_get_req_id(request) if request else None,
        inputs={"time": time, "force": force},
        lineage={"sensor": sensor_used, "source_url": url_used, "local_file": resp["file"]},
    )
    try:
        await _wxg_log_payload(resp["governance"], output={"status": resp["status"], "file": resp["file"]})
    except Exception:
        pass
    return resp


@app.get("/seaice/point")
async def point_sample(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    time: Optional[str] = None,
    request: Request = None,
):
    t = time or datetime.utcnow().strftime("%Y-%m-%d")
    local_path, sensor, _ = _ensure_local_file(t, force=False)
    ds = _open_ds(local_path)

    x, y = _to_proj(lon, lat)
    sel = _nearest_index(ds, x, y, lon=lon, lat=lat)

    if NSIDC_VAR_NAME not in ds.variables:
        raise HTTPException(status_code=500, detail=f"Variable '{NSIDC_VAR_NAME}' not found in dataset.")

    da = ds[NSIDC_VAR_NAME]
    val = float(da.isel(**sel).values)

    scaled = False
    vf = val
    # Heuristic: many sea-ice conc products are 0..1 or 0..100
    if vf > 1.5:
        vf = vf / 100.0
        scaled = True

    resp = {
        "lat": lat,
        "lon": lon,
        "time": t,
        "value_raw": float(val),
        "value_fraction": float(vf),
        "file": os.path.abspath(local_path),
        "variable": NSIDC_VAR_NAME,
        "projection": NSIDC_CRS,
        "sensor": sensor,
    }
    resp["governance"] = _gov_base(
        endpoint="/seaice/point",
        request_id=_get_req_id(request) if request else None,
        inputs={"lat": lat, "lon": lon, "time": t},
        lineage={"local_file": resp["file"], "variable": NSIDC_VAR_NAME, "crs": NSIDC_CRS, "sensor": sensor},
        quality={"scaling_applied": scaled, "nearest_index": True},
    )
    try:
        await _wxg_log_payload(resp["governance"], output={"value_fraction": resp["value_fraction"]})
    except Exception:
        pass
    return resp


@app.post("/seaice/stats")
async def bbox_stats(payload: Dict[str, Any] = Body(...), request: Request = None):
    """
    Payload example:
    {
      "bbox": [minLat, minLon, maxLat, maxLon],   # WGS84
      "time": "YYYY-MM-DD"
    }
    """
    bbox = payload.get("bbox")
    t = payload.get("time")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise HTTPException(status_code=400, detail="bbox must be a 4-element list [minLat, minLon, maxLat, maxLon].")
    try:
        minLat, minLon, maxLat, maxLon = [float(v) for v in bbox]
    except Exception:
        raise HTTPException(status_code=400, detail="bbox elements must be numeric.")

    if t is None:
        t = datetime.utcnow().strftime("%Y-%m-%d")

    local_path, sensor, _ = _ensure_local_file(t, force=False)
    ds = _open_ds(local_path)

    if NSIDC_VAR_NAME not in ds.variables:
        raise HTTPException(status_code=500, detail=f"Variable '{NSIDC_VAR_NAME}' not found in dataset.")

    # Project bbox corners to target CRS
    x0, y0 = _to_proj(minLon, minLat)
    x1, y1 = _to_proj(maxLon, maxLat)
    xmin, xmax = (min(x0, x1), max(x0, x1))
    ymin, ymax = (min(y0, y1), max(y0, y1))

    # Find nearest index ranges
    def _idx(coord_vals, a, b):
        i0 = int(np.argmin(np.abs(coord_vals - a)))
        i1 = int(np.argmin(np.abs(coord_vals - b)))
        return (min(i0, i1), max(i0, i1))

    # Resolve coordinate names
    x_name = next((c for c in ["x", "projection_x_coordinate", "lon_east", "xc", "cols"] if c in ds.coords), None)
    y_name = next((c for c in ["y", "projection_y_coordinate", "lat_north", "yc", "rows"] if c in ds.coords), None)
    if x_name is None or y_name is None:
        x_name = x_name or "x"
        y_name = y_name or "y"
        if x_name not in ds.dims or y_name not in ds.dims:
            raise HTTPException(status_code=500, detail="Dataset missing recognizable x/y coordinates.")

    xs = ds.coords[x_name].values
    ys = ds.coords[y_name].values

    ix0, ix1 = _idx(xs, xmin, xmax)
    iy0, iy1 = _idx(ys, ymin, ymax)

    da = ds[NSIDC_VAR_NAME].isel({x_name: slice(ix0, ix1 + 1), y_name: slice(iy0, iy1 + 1)}).values
    vals = np.array(da, dtype=float)

    if float(np.nanmax(vals)) > 1.5:
        vals = vals / 100.0
        scaled = True
    else:
        scaled = False

    resp = {
        "bbox": [minLat, minLon, maxLat, maxLon],
        "time": t,
        "count": int(np.isfinite(vals).sum()),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "units": "fraction",
        "file": os.path.abspath(local_path),
        "variable": NSIDC_VAR_NAME,
        "projection": NSIDC_CRS,
        "sensor": sensor,
    }
    resp["governance"] = _gov_base(
        endpoint="/seaice/stats",
        request_id=_get_req_id(request) if request else None,
        inputs={"bbox": bbox, "time": t},
        lineage={"local_file": resp["file"], "variable": NSIDC_VAR_NAME, "crs": NSIDC_CRS, "sensor": sensor},
        quality={"scaling_applied": scaled, "aggregation": ["min", "max", "mean", "median"]},
    )
    try:
        await _wxg_log_payload(resp["governance"], output={"mean": resp["mean"], "count": resp["count"]})
    except Exception:
        pass
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("seaice_agent:app", host="0.0.0.0", port=8090, reload=False)