# ais_agent.py  (v0.2.0)
# Minimal AIS REST→REST gateway with AOI registry + governance meta (FQ URLs)
# Run:  uvicorn ais_agent:app --host 0.0.0.0 --port 8100 --reload
# Deps: pip install fastapi uvicorn httpx pydantic

from __future__ import annotations
import os, json, math, hashlib, logging, sys, time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from ais_vessel_info import fetch_vessel_info_by_imo, fetch_vessel_info_by_mmsi, fetch_vessel_info_by_name

import httpx
from fastapi import FastAPI, Query, Header, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_JSON  = os.getenv("LOG_JSON", "1") in {"1", "true", "True"}

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)
logger.info("***** Starting logger...")

APP_NAME                = "ais_agent"
APP_VERSION             = "0.2.0"
UPSTREAM_BASE           = os.getenv("AIS_UPSTREAM_BASE", "https://services.marinetraffic.com/api")
UPSTREAM_BASE_GRAPHQL   = os.getenv("AIS_UPSTREAM_BASE_GRAPHQL", "https://api.kpler.marinetraffic.com/v2/vessels/graphql")
AOI_PATH                = os.getenv("AOI_GEOJSON_PATH", "alaska_uscg_arctic_aois.geojson")

ENV_KEY                     = os.getenv("AIS_EXPORTVESSELS_KEY", "")
AIS_EXPORTVESSELS_KEY       = os.getenv("AIS_EXPORTVESSELS_KEY", "")
AIS_SHIPSEARCH_KEY          = os.getenv("AIS_SHIPSEARCH_KEY", "")
AIS_VESSELPHOTO_KEY         = os.getenv("AIS_VESSELPHOTO_KEY", "")
AIS_PORTCALLS_KEY           = os.getenv("AIS_PORTCALLS_KEY", "")
AIS_VESSELEVENTS_KEY        = os.getenv("AIS_VESSELEVENTS_KEY", "")
AIS_OWNERSHIP_KEY           = os.getenv("AIS_OWNERSHIP_KEY", "")

# ----- Ship types (per swagger.json) -----
SHIPTYPE_CODE_SET = {2, 4, 6, 7, 8}
SHIPTYPE_NAME_TO_CODE = {
    "fishing": 2,
    "high_speed": 4,
    "high speed craft": 4,
    "passenger": 6,
    "cargo": 7,
    "tanker": 8,
}
AOI_MSGTYPES = {"simple", "extended", "full"}

SENSITIVE_KEYS_DEFAULT = {"apikey", "api_key", "token", "auth", "authorization"}

# ----- Models -----
class GovernanceMeta(BaseModel):
    source: str
    endpoint: str
    variablesHash: Optional[str] = None
    fetchedAt: str
    version: str
    upstreamEndpoint: Optional[str] = None
    aoiEndpoint: Optional[str] = None
    aoiId: Optional[str] = None
    aoiHash: Optional[str] = None
    bbox: Optional[List[float]] = None
    registryHash: Optional[str] = None
    aoiSource: Optional[str] = None

# ----- Helpers -----
def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256_hex(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()

def canonical_json(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")

def vhash(d: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json(d))

def resolve_api_key(header_key: Optional[str]) -> str:
    key = header_key or ENV_KEY
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key. Provide X-Upstream-Api-Key or set AIS_KEY.")
    return key

async def upstream_get(path: str, params: Dict[str, Any]) -> Dict[str, Any] | str:
    url = f"{UPSTREAM_BASE}{path}"

    logger.info(f"upstream_get {url} {params}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url, params=params)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail={"upstream_error": r.text})
        ctype = r.headers.get("content-type", "")
        return r.json() if "application/json" in ctype else r.text

async def upstream2_get(path: str, params: Dict[str, Any]) -> Dict[str, Any] | str:
    url = f"{UPSTREAM_BASE_GRAPHQL}{path}"

    logger.info(f"upstream2_get {url} {params}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url, params=params)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail={"upstream_error": r.text})
        ctype = r.headers.get("content-type", "")
        return r.json() if "application/json" in ctype else r.text


def bbox_around_point_nm(lon: float, lat: float, radius_nm: float) -> Tuple[float, float, float, float]:
    radius_km = radius_nm * 1.852
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.1, math.cos(math.radians(lat))))
    minlon = round(lon-dlon,2)
    minlat = round(lat - dlat,2)
    maxlon = round(lon+dlon,2)
    maxlat = round(lat+dlat,2)

    if minlon < -180:

        minlon += 360
        t = maxlon
        maxlon = minlon
        minlon = t

    return (minlon, minlat, maxlon, maxlat)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    Great-circle distance between two WGS84 lat/lon points (degrees), in kilometers.
    Numerically stable haversine using atan2 and clamping.

    Args:
        lat1, lon1: Point 1 latitude/longitude in degrees
        lat2, lon2: Point 2 latitude/longitude in degrees

    Returns:
        Distance in kilometers (float)
    """
    # Mean Earth radius per IUGG
    R = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    sin_dphi = math.sin(dphi * 0.5)
    sin_dlambda = math.sin(dlambda * 0.5)

    a = sin_dphi * sin_dphi + math.cos(phi1) * math.cos(phi2) * sin_dlambda * sin_dlambda
    # Clamp for floating point safety
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)

    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def extract_lat_lon(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lower = {k.lower(): v for k, v in rec.items()} if isinstance(rec, dict) else {}
    lat = lower.get("lat") or lower.get("latitude") or lower.get("y")
    lon = lower.get("lon") or lower.get("long") or lower.get("longitude") or lower.get("x")
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
    except Exception:
        lat, lon = None, None
    return lat, lon

def normalize_shiptype(shiptype: Optional[str]) -> Optional[int]:
    if not shiptype:
        return None
    s = shiptype.strip().lower()
    if s.isdigit():
        code = int(s)
        if code not in SHIPTYPE_CODE_SET:
            raise HTTPException(status_code=400, detail=f"Invalid shiptype code. Allowed: {sorted(SHIPTYPE_CODE_SET)}")
        return code
    if s not in SHIPTYPE_NAME_TO_CODE:
        raise HTTPException(status_code=400, detail=f"Invalid shiptype name. Allowed: {sorted(SHIPTYPE_NAME_TO_CODE.keys())}")
    return SHIPTYPE_NAME_TO_CODE[s]

def make_identifier_params(ship_id: Optional[str], mmsi: Optional[str], imo: Optional[str]) -> Dict[str, Any]:
    ids = [x for x in [ship_id, mmsi, imo] if x]
    if len(ids) != 1:
        raise HTTPException(status_code=400, detail="Provide exactly ONE of: ship_id, mmsi, or imo.")
    params: Dict[str, Any] = {}
    if ship_id: params["shipid"] = ship_id   # upstream key is 'shipid'
    if mmsi:    params["mmsi"] = mmsi
    if imo:     params["imo"] = imo
    return params

def fq(
    request: Request,
    sanitize_keys: set[str] | None = None,
    overrides: dict[str, str | int | float] | None = None,
    include_existing: bool = True,
) -> str:
    """
    Build a fully-qualified URL for the incoming request.

    - include_existing: keep the request's current query parameters
    - sanitize_keys: remove these query keys (case-insensitive)
    - overrides: add/override these query parameters
    """
    sanitize = {k.lower() for k in (sanitize_keys or SENSITIVE_KEYS_DEFAULT)}

    # Start from current query params (optional)
    if include_existing:
        items = [(k, v) for k, v in request.query_params.multi_items()
                 if k.lower() not in sanitize]
        qp = dict(items)
    else:
        qp = {}

    # Apply overrides (also sanitized)
    if overrides:
        for k, v in overrides.items():
            if k.lower() not in sanitize:
                qp[k] = v

    # Build the URL
    if qp:
        url = request.url.replace_query_params(**qp)
    else:
        # Clear all query params (don't use remove_query_params() with no keys)
        url = request.url.replace(query="")

    return str(url)

def upstream_template(path: str) -> str:
    # Show the upstream endpoint template without secrets
    base = UPSTREAM_BASE.rstrip("/")
    return f"{base}{path}"

# ----- AOI registry -----
class AoiFeature(BaseModel):
    type: str
    properties: Dict[str, Any]
    geometry: Dict[str, Any]

class AoiRegistry:
    def __init__(self, path: str):
        self.path = path
        self._features: Dict[str, AoiFeature] = {}
        self._registry_hash: str = ""
        self._raw_geojson: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            self._features = {}
            self._registry_hash = ""
            self._raw_geojson = {}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        self._registry_hash = sha256_hex(raw_text.encode("utf-8"))
        self._raw_geojson = json.loads(raw_text)
        feats = self._raw_geojson.get("features", [])
        self._features = {}
        for feat in feats:
            props = (feat.get("properties") or {})
            fid = props.get("id") or props.get("name")
            if not fid:
                continue
            self._features[fid] = AoiFeature(type=feat["type"], properties=props, geometry=feat.get("geometry") or {})

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for fid, f in self._features.items():
            out.append({
                "id": fid,
                "display_name": f.properties.get("display_name", fid),
                "type": f.properties.get("type", "bbox"),
                "bbox": f.properties.get("bbox"),
            })
        return out

    def get(self, fid: str) -> AoiFeature:
        if fid not in self._features:
            raise HTTPException(status_code=404, detail=f"AOI '{fid}' not found")
        return self._features[fid]

    def bbox_for(self, fid: str) -> Tuple[float, float, float, float]:
        feat = self.get(fid)
        if feat.properties.get("type") != "bbox":
            raise HTTPException(status_code=400, detail=f"AOI '{fid}' is not a bbox type")
        bbox = feat.properties.get("bbox")
        if not bbox or len(bbox) != 4:
            raise HTTPException(status_code=400, detail=f"AOI '{fid}' has no valid bbox")
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

AOI = AoiRegistry(AOI_PATH)

# ----- FastAPI app -----
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="AIS gateway with AOI registry and governance meta (FQ URLs).",
)

# Health
@app.get("/ais/health", tags=["Health"])
async def health(request: Request):
    ok, detail = True, "ok"
    logger.info(f"health v:{APP_VERSION} b:{UPSTREAM_BASE} n:{APP_NAME}")
    return JSONResponse({
        "status": "ok" if ok else "degraded",
        "version": APP_VERSION,
        "upstream": UPSTREAM_BASE,
        "detail": detail,
        "meta": GovernanceMeta(
            source=APP_NAME,
            endpoint=fq(request),
            fetchedAt=now_iso(),
            version=APP_VERSION)
    })

# ----- AOI endpoints -----
@app.get("/ais/aoi", tags=["AOI"])
async def list_aois(request: Request):
    return JSONResponse({
        "items": AOI.list(),
        "meta": {
            "endpoint": fq(request),
            "source": AOI_PATH,
            "fetchedAt": now_iso(),
            "registryHash": AOI._registry_hash,
            "version": APP_VERSION,
        },
    })

@app.get("/ais/aoi/{aoi_id}", tags=["AOI"])
async def get_aoi(aoi_id: str = Path(..., description="AOI identifier"), request: Request = None):
    feat = AOI.get(aoi_id)
    bbox = feat.properties.get("bbox")
    aoi_hash = sha256_hex(canonical_json({"properties": feat.properties, "geometry": feat.geometry}))
    return JSONResponse({
        "feature": feat.dict(),
        "meta": {
            "endpoint": fq(request),
            "aoiId": aoi_id,
            "aoiHash": aoi_hash,
            "bbox": bbox,
            "source": AOI_PATH,
            "fetchedAt": now_iso(),
            "registryHash": AOI._registry_hash,
            "version": APP_VERSION,
        },
    })

# ----- Vessel search -----
@app.get("/ais/vessels/search", tags=["Vessels"])
async def vessels_search(
    request: Request,
    shipname: Optional[str] = Query(None, description="Vessel name (full or partial)"),
    mmsi: Optional[str] = Query(None, description="Maritime Mobile Service Identity"),
    imo: Optional[str] = Query(None, description="IMO number")
):
    apikey = AIS_SHIPSEARCH_KEY
    params: Dict[str, Any] = {}
    if shipname: params["shipname"] = shipname
    if mmsi:     params["mmsi"] = mmsi
    if imo:      params["imo"] = imo
    params['protocol'] = 'jsono'
    payload = await upstream_get(f"/shipsearch/{apikey}", params)
    meta = GovernanceMeta(
        source=APP_NAME,
        endpoint=fq(request),
        upstreamEndpoint=upstream_template("/shipsearch/{api_key}"),
        variablesHash=vhash(params),
        fetchedAt=now_iso(),
        version=APP_VERSION,
    )
    return JSONResponse({"nodes": payload, "meta": meta.dict()})

# ----- Vessels in AOI -----
@app.get("/ais/vessels/aoi", tags=["Vessels"])
async def vessels_in_aoi(
    request: Request,
    aoi_id: Optional[str] = Query(None, description="Registered AOI id; alternative to bbox"),
    bbox: Optional[str]   = Query(None, description="minLon,minLat,maxLon,maxLat (WGS84)"),
    timespan: Optional[int] = Query(60, description="Minutes back"),
    shiptype: Optional[str] = Query(None, description="2=fishing, 4=high_speed, 6=passenger, 7=cargo, 8=tanker"),
    msgtype: str = Query("simple", description="simple | extended | full")
):
    apikey = AIS_EXPORTVESSELS_KEY
    if msgtype not in AOI_MSGTYPES:
        raise HTTPException(status_code=400, detail=f"Invalid msgtype. Allowed: {sorted(AOI_MSGTYPES)}")

    if aoi_id:
        minLon, minLat, maxLon, maxLat = AOI.bbox_for(aoi_id)
    elif bbox:
        try:
            minLon, minLat, maxLon, maxLat = [float(x.strip()) for x in bbox.split(",")]
        except Exception:
            raise HTTPException(status_code=400, detail="bbox must be 'minLon,minLat,maxLon,maxLat'")
    else:
        raise HTTPException(status_code=400, detail="Provide either aoi_id or bbox")

    shiptype_code = normalize_shiptype(shiptype)
    params = {"minlat": minLat, "minlon": minLon, "maxlat": maxLat, "maxlon": maxLon, "msgtype": msgtype}
    params['protocol'] = 'jsono'
    params['v'] = 8
    
    if timespan is not None: params["timespan"] = timespan
    if shiptype_code is not None: params["shiptype"] = shiptype_code

    payload = await upstream_get(f"/exportvessels/{apikey}", params)

    meta_dict = {
        "source": APP_NAME,
        "endpoint": fq(request),
        "upstreamEndpoint": upstream_template("/exportvessels/{api_key}"),
        "variablesHash": vhash({**params, "aoi_id": aoi_id} if aoi_id else params),
        "fetchedAt": now_iso(),
        "version": APP_VERSION,
    }
    if aoi_id:
        aoi_feat = AOI.get(aoi_id)
        aoi_hash = sha256_hex(canonical_json({"properties": aoi_feat.properties, "geometry": aoi_feat.geometry}))
        meta_dict.update({
            "aoiEndpoint": fq(request, include_existing=False).replace("/vessels/aoi", f"/aoi/{aoi_id}"),
            "aoiId": aoi_id,
            "aoiHash": aoi_hash,
            "bbox": [minLon, minLat, maxLon, maxLat],
            "registryHash": AOI._registry_hash,
            "aoiSource": AOI_PATH,
        })
    return JSONResponse({"nodes": payload, "meta": meta_dict})

# ----- Vessels nearby -----
@app.get("/ais/vessels/nearby", tags=["Vessels"])
async def vessels_nearby(
    request: Request,
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Latitude (WGS84)"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Longitude (WGS84)"),
    radius_nm: float = Query(50.0, gt=0, le=1000, description="Radius in nautical miles"),
    aoi_id: Optional[str] = Query(None, description="If provided (bbox AOI), its centroid is used"),
    timespan: Optional[int] = Query(60, description="Minutes back"),
    shiptype: Optional[str] = Query(None, description="2=fishing, 4=high_speed, 6=passenger, 7=cargo, 8=tanker"),
    msgtype: str = Query("simple", description="simple | extended | full")
):
    apikey = AIS_EXPORTVESSELS_KEY
    if msgtype not in AOI_MSGTYPES:
        raise HTTPException(status_code=400, detail=f"Invalid msgtype. Allowed: {sorted(AOI_MSGTYPES)}")

    if aoi_id:
        minLon0, minLat0, maxLon0, maxLat0 = AOI.bbox_for(aoi_id)
        lon = (minLon0 + maxLon0) / 2.0
        lat = (minLat0 + maxLat0) / 2.0
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Provide lat & lon or aoi_id")

    shiptype_code = normalize_shiptype(shiptype)

    minLon, minLat, maxLon, maxLat = bbox_around_point_nm(lon, lat, radius_nm)
    params = {"minlat": minLat, "minlon": minLon, "maxlat": maxLat, "maxlon": maxLon, "msgtype": msgtype}
    params['protocol'] = 'jsono'
    params['v'] = 8
    
    if timespan is not None: params["timespan"] = timespan
    if shiptype_code is not None: params["shiptype"] = shiptype_code

    upstream = await upstream_get(f"/exportvessels/{apikey}", params)

    #logger.info(f"nearvy, {upstream}")
    radius_km = radius_nm * 1.852
    if isinstance(upstream, list):
        filtered = []
        for rec in upstream:
            plat, plon = extract_lat_lon(rec)
            if plat is None or plon is None:
                continue
            distance = haversine_km(lat, lon, plat, plon) 
            if  distance <= radius_km:
                filtered.append(rec)
            else:
                logger.info(f'failed: {lat} {lon} {plat} {plon} d:{distance} r:{radius_km}')
        nodes = filtered
    else:
        nodes = upstream

    meta_dict = {
        "source": APP_NAME,
        "endpoint": fq(request),
        "upstreamEndpoint": upstream_template("/exportvessels/{api_key}"),
        "variablesHash": vhash({**params, "lat": lat, "lon": lon, "radius_nm": radius_nm, "aoi_id": aoi_id}),
        "fetchedAt": now_iso(),
        "version": APP_VERSION,
    }
    if aoi_id:
        aoi_feat = AOI.get(aoi_id)
        aoi_hash = sha256_hex(canonical_json({"properties": aoi_feat.properties, "geometry": aoi_feat.geometry}))
        meta_dict.update({
            "aoiEndpoint": fq(request, include_existing=False).replace("/vessels/nearby", f"/aoi/{aoi_id}"),
            "aoiId": aoi_id,
            "aoiHash": aoi_hash,
            "bbox": [minLon, minLat, maxLon, maxLat],
            "registryHash": AOI._registry_hash,
            "aoiSource": AOI_PATH,
        })
    return JSONResponse({"nodes": nodes, "meta": meta_dict})

# ----- Vessel info -----
@app.get("/ais/vessel/info", tags=["Vessels"])
async def vessel_info(
    request: Request,
    mmsi: Optional[str]     = Query(None, description="Maritime Mobile Service Identity"),
    imo: Optional[str]      = Query(None, description="IMO number"),
    shipname: Optional[str]     = Query(None, description="Ship Name"),
):
    if imo:
        payload = fetch_vessel_info_by_imo(imo, after_cursor=None)
    if mmsi:
        payload = fetch_vessel_info_by_mmsi(mmsi, after_cursor=None)
    if shipname:
        payload = fetch_vessel_info_by_name(shipname, after_cursor=None)

    meta = GovernanceMeta(
        source=APP_NAME,
        endpoint=fq(request),
        fetchedAt=now_iso(),
        version=APP_VERSION,
    )
    nodes = payload['data']['vessels']['nodes']
    return JSONResponse({"nodes":nodes, "meta": meta.dict()})

# ----- Vessel info -----
@app.get("/ais//vessel/photo", tags=["Vessels"])
async def vessel_photo(
    request: Request,
    ship_id: Optional[str] = Query(None, description="Provide vessel id"),
    mmsi: Optional[str]   = Query(None, description="Maritime Mobile Service Identity"),
    imo: Optional[str]    = Query(None, description="IMO number")
):
    apikey = AIS_VESSELPHOTO_KEY
    id_params = make_identifier_params(ship_id,mmsi,imo)
    if mmsi:
        id_params['vessel_id'] = mmsi
    if imo:
        id_params['vessel_id'] = imo

    id_params['protocol'] = 'jsono'
    payload = await upstream_get(f"/exportvesselphoto/{apikey}", id_params)
    meta = GovernanceMeta(
        source=APP_NAME,
        endpoint=fq(request),
        upstreamEndpoint=upstream_template("/exportvesselphoto/{api_key}"),
        variablesHash=vhash(id_params),
        fetchedAt=now_iso(),
        version=APP_VERSION,
    )
    return JSONResponse({"node": payload, "meta": meta.dict()})

# ----- Vessel track -----
@app.get("/ais/vessel/track", tags=["Tracks"])
async def vessel_track(
    request: Request,
    ship_id: Optional[str] = Query(None, description="Provider vessel id"),
    mmsi: Optional[str]    = Query(None, description="Maritime Mobile Service Identity"),
    imo: Optional[str]     = Query(None, description="IMO number"),
    fromdt: Optional[str]  = Query(None, description="UTC start, e.g., 2025-09-01 00:00"),
    todt: Optional[str]    = Query(None, description="UTC end, e.g., 2025-09-02 00:00"),
    timespan: Optional[int]= Query(None, description="Minutes back (alternative to from/todt)")
):
    apikey = AIS_EXPORTVESSELS_KEY
    params = make_identifier_params(ship_id, mmsi, imo)
    params['protocol'] = 'json'
    params['v'] = 3

    if fromdt:   params["from"] = fromdt
    if todt:     params["to"]   = todt
    if timespan: params["timespan"] = timespan
    payload = await upstream_get(f"/exportvesseltrack/{apikey}", params)
    meta = GovernanceMeta(
        source=APP_NAME,
        endpoint=fq(request),
        upstreamEndpoint=upstream_template("/exportvesseltrack/{api_key}"),
        variablesHash=vhash(params),
        fetchedAt=now_iso(),
        version=APP_VERSION,
    )
    return JSONResponse({"nodes": payload, "meta": meta.dict()})
