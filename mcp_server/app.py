"""MCP bridge app that re-uses the AIS agent implementation.

This module defines all FastAPI endpoints on the local `app` and delegates
their implementation to functions in the bundled `ais_agent.ais_agent`
module under this directory. All routing lives here; the ais_agent module
is logic-only.
"""
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Query, Path as FPath
from fastapi.responses import FileResponse, JSONResponse

from ais_agent import ais_agent

logger = logging.getLogger("mcp_server")
logging.basicConfig(level=logging.INFO)

HERE = Path(__file__).resolve().parent
OPENAPI_YAML = HERE / "ais_openapi.yaml"

# Bridge FastAPI app: docs and openapi are served under /mcp/*
app = FastAPI(
    title="MCP Bridge",
    version="0.1",
    docs_url="/mcp/docs",
    redoc_url="/mcp/redoc",
    openapi_url="/mcp/openapi.json",
)


@app.get("/mcp/openapi.yaml")
async def openapi_yaml():
    """Serve the curated AIS OpenAPI YAML used by orchestrators such as watsonx."""
    if OPENAPI_YAML.exists():
        return FileResponse(str(OPENAPI_YAML), media_type="application/yaml")
    return JSONResponse({"error": "openapi.yaml not found"}, status_code=404)

# ----- Health -----
@app.get("/health", tags=["Health"])
async def health_root(request: Request):
    return await ais_agent.health(request)


@app.get("/mcp/ais/health", tags=["Health"])
async def health(request: Request):
    return await ais_agent.health(request)


# ----- AOI endpoints -----
@app.get("/mcp/ais/aoi", tags=["AOI"])
async def list_aois(request: Request):
    return await ais_agent.list_aois(request)


@app.get("/mcp/ais/aoi/{aoi_id}", tags=["AOI"])
async def get_aoi(
    aoi_id: str = FPath(..., description="AOI identifier"),
    request: Request = None,
):
    return await ais_agent.get_aoi(aoi_id, request)


# ----- Vessels in AOI -----
@app.get("/mcp/ais/vessels/aoi", tags=["Vessels"])
async def vessels_in_aoi(
    request: Request,
    aoi_id: Optional[str] = Query(None, description="Registered AOI id; alternative to bbox"),
    bbox: Optional[str] = Query(None, description="minLon,minLat,maxLon,maxLat (WGS84)"),
    timespan: Optional[int] = Query(60, description="Minutes back"),
    shiptype: Optional[str] = Query(None, description="2=fishing, 4=high_speed, 6=passenger, 7=cargo, 8=tanker"),
    msgtype: str = Query("simple", description="simple | extended | full"),
):
    return await ais_agent.vessels_in_aoi(request, aoi_id, bbox, timespan, shiptype, msgtype)


# ----- Vessels nearby -----
@app.get("/mcp/ais/vessels/nearby", tags=["Vessels"])
async def vessels_nearby(
    request: Request,
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Latitude (WGS84)"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Longitude (WGS84)"),
    radius_nm: float = Query(50.0, gt=0, le=1000, description="Radius in nautical miles"),
    aoi_id: Optional[str] = Query(None, description="If provided (bbox AOI), its centroid is used"),
    timespan: Optional[int] = Query(60, description="Minutes back"),
    shiptype: Optional[str] = Query(None, description="2=fishing, 4=high_speed, 6=passenger, 7=cargo, 8=tanker"),
    msgtype: str = Query("simple", description="simple | extended | full"),
):
    return await ais_agent.vessels_nearby(
        request, lat, lon, radius_nm, aoi_id, timespan, shiptype, msgtype
    )


# ----- Vessel info -----
@app.get("/mcp/ais/vessel/info", tags=["Vessels"])
async def vessel_info(
    request: Request,
    mmsi: Optional[str] = Query(None, description="Maritime Mobile Service Identity"),
    imo: Optional[str] = Query(None, description="IMO number"),
    shipname: Optional[str] = Query(None, description="Ship Name"),
):
    return await ais_agent.vessel_info(request, mmsi, imo, shipname)


# ----- Vessel photo -----
@app.get("/mcp/ais/vessel/photo", tags=["Vessels"])
async def vessel_photo(
    request: Request,
    ship_id: Optional[str] = Query(None, description="Provide vessel id"),
    mmsi: Optional[str] = Query(None, description="Maritime Mobile Service Identity"),
    imo: Optional[str] = Query(None, description="IMO number"),
):
    return await ais_agent.vessel_photo(request, ship_id, mmsi, imo)


# ----- Vessel track -----
@app.get("/mcp/ais/vessel/track", tags=["Tracks"])
async def vessel_track(
    request: Request,
    ship_id: Optional[str] = Query(None, description="Provider vessel id"),
    mmsi: Optional[str] = Query(None, description="Maritime Mobile Service Identity"),
    imo: Optional[str] = Query(None, description="IMO number"),
    fromdate: Optional[str] = Query(None, description="UTC start, e.g., 2025-09-01 00:00"),
    todate: Optional[str] = Query(None, description="UTC end, e.g., 2025-09-02 00:00"),
    days: Optional[int] = Query(
        None, description="The number of days, starting from the time of request and going backwards"
    ),
):
    return await ais_agent.vessel_track(request, ship_id, mmsi, imo, fromdate, todate, days)


# ----- Vessel Events -----
@app.get("/mcp/ais/vessel/events", tags=["Events"])
async def vessel_events(
    request: Request,
    ship_id: Optional[str] = Query(None, description="Provider vessel id"),
    mmsi: Optional[str] = Query(None, description="Maritime Mobile Service Identity"),
    imo: Optional[str] = Query(None, description="IMO number"),
    fromdate: Optional[str] = Query(None, description="UTC start, e.g., 2025-09-01 00:00"),
    todate: Optional[str] = Query(None, description="UTC end, e.g., 2025-09-02 00:00"),
    timespan: Optional[int] = Query(
        None,
        description="The maximum age, in minutes, of the returned port calls. Maximum value is 2880",
    ),
):
    return await ais_agent.vessel_events(
        request, ship_id, mmsi, imo, fromdate, todate, timespan
    )


# ----- Single Vessel Portcalls -----
@app.get("/mcp/ais/vessel/portcalls", tags=["PortCalls"])
async def vessel_portcalls(
    request: Request,
    ship_id: Optional[str] = Query(None, description="Provider vessel id"),
    fromdate: Optional[str] = Query(None, description="UTC start, e.g., 2025-09-01 00:00"),
    todate: Optional[str] = Query(None, description="UTC end, e.g., 2025-09-02 00:00"),
    timespan: Optional[int] = Query(
        None,
        description="The maximum age, in minutes, of the returned port calls. Maximum value is 2880",
    ),
):
    return await ais_agent.vessel_portcalls(
        request, ship_id, fromdate, todate, timespan
    )


# ----- Portcalls -----
@app.get("/mcp/ais/portcalls", tags=["PortCalls"])
async def portcalls(
    request: Request,
    port_id: Optional[str] = Query(None, description="Port id or UN/LOCODE"),
    fromdate: Optional[str] = Query(None, description="UTC start, e.g., 2025-09-01 00:00"),
    todate: Optional[str] = Query(None, description="UTC end, e.g., 2025-09-02 00:00"),
    timespan: Optional[int] = Query(
        None,
        description="The maximum age, in minutes, of the returned port calls. Maximum value is 2880",
    ),
):
    return await ais_agent.portcalls(
        request, port_id, fromdate, todate, timespan
    )


# ----- Routing -----
@app.get("/mcp/ais/routing/distance_to_port", tags=["Routing"])
async def distance_to_port(
    request: Request,
    start_port: Optional[str] = Query(None, description="Starting Port UN/LOCODE"),
    end_port: Optional[str] = Query(None, description="Ending Port UN/LOCODE"),
):
    return await ais_agent.distance_to_port(request, start_port, end_port)


@app.get("/mcp/ais/routing/vessel_route_to_port", tags=["Routing"])
async def vessel_route_to_port(
    request: Request,
    ship_id: Optional[str] = Query(None, description="Provider vessel id"),
    imo: Optional[str] = Query(None, description="Provider vessel imo"),
    mmsi: Optional[str] = Query(None, description="Provider vessel mmsi"),
    port_id: Optional[str] = Query(None, description="Ending Port UN/LOCODE"),
):
    return await ais_agent.vessel_route_to_port(
        request, ship_id, imo, mmsi, port_id
    )

