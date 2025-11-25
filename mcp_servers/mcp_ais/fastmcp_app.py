"""
FastMCP server exposing AIS tools as MCP tools.

Author: Patrice G. Cappelaere, IBM Federal

This module defines a small FastMCP application that wraps selected AIS agent
capabilities as MCP tools. It is analogous in structure to the METOC FastMCP
server under `mcp_servers/mcp_metoc/app.py`.
"""

from __future__ import annotations

import os
from typing import Optional

from fastmcp import FastMCP  # type: ignore

import ais_agent as core

mcp = FastMCP(name="AIS MCP Server")


@mcp.tool
async def get_ais_health() -> dict:
    """
    Simple health probe for the AIS MCP server.

    Returns:
        dict: Basic health status and upstream base URL.
    """
    
    response = {
        "status": "ok",
        "provider": "ais",
        "upstream_base": core.UPSTREAM_BASE,
        "version": core.APP_VERSION,
    }

    return response


@mcp.tool
async def list_ais_aois() -> dict:
    """
    List Areas of Interest (AOIs) known to the AIS agent.

    Returns:
        dict: AOI items and governance metadata.
    """
   
    items = core.AOI.list()
    response = {
        "items": items,
        "meta": {
            "endpoint": "list_ais_aois",
            "source": core.AOI_PATH,
            "fetchedAt": core.now_iso(),
            "registryHash": core.AOI._registry_hash,
            "version": core.APP_VERSION,
        },
    }
    return response


@mcp.tool
async def get_ais_aoi(aoi_id: str) -> dict:
    """
    Retrieve a specific AOI by identifier.

    Args:
        aoi_id: AOI identifier string.

    Returns:
        dict: AOI feature and governance metadata.
    """
    feat = core.AOI.get(aoi_id)
    bbox = feat.properties.get("bbox")
    aoi_hash = core.sha256_hex(
        core.canonical_json({"properties": feat.properties, "geometry": feat.geometry})
    )
    response = {
        "feature": feat.dict(),
        "meta": {
            "endpoint": "get_ais_aoi",
            "aoiId": aoi_id,
            "aoiHash": aoi_hash,
            "bbox": bbox,
            "source": core.AOI_PATH,
            "fetchedAt": core.now_iso(),
            "registryHash": core.AOI._registry_hash,
            "version": core.APP_VERSION,
        },
    }
    return response


@mcp.tool
async def ais_vessels_in_aoi(
    aoi_id: Optional[str] = None,
    bbox: Optional[str] = None,
    timespan: Optional[int] = 60,
    shiptype: Optional[str] = None,
    msgtype: str = "simple",
) -> dict:
    """
    List vessels inside a named AOI or bounding box.

    Args:
        aoi_id: Registered AOI id; alternative to bbox.
        bbox: Bounding box as 'minLon,minLat,maxLon,maxLat' (WGS84).
        timespan: Minutes back from now to consider.
        shiptype: Optional AIS ship type filter.
        msgtype: Detail level of AIS messages: simple | extended | full.

    Returns:
        dict: Vessel list payload and governance metadata.
    """

    apikey = core.AIS_EXPORTVESSELS_KEY
    if msgtype not in core.AOI_MSGTYPES:
        raise core.HTTPException(
            status_code=400, detail=f"Invalid msgtype. Allowed: {sorted(core.AOI_MSGTYPES)}"
        )

    if aoi_id:
        minLon, minLat, maxLon, maxLat = core.AOI.bbox_for(aoi_id)
    elif bbox:
        try:
            minLon, minLat, maxLon, maxLat = [float(x.strip()) for x in bbox.split(",")]
        except Exception as exc:  # pragma: no cover - input validation
            raise core.HTTPException(
                status_code=400,
                detail="bbox must be 'minLon,minLat,maxLon,maxLat'",
            ) from exc
    else:
        raise core.HTTPException(status_code=400, detail="Provide either aoi_id or bbox")

    try:
        shiptype_code = core.normalize_shiptype(shiptype)
    except e:
        shiptype_code = None
    params = {
        "minlat": minLat,
        "minlon": minLon,
        "maxlat": maxLat,
        "maxlon": maxLon,
        "msgtype": msgtype,
        "protocol": "jsono",
        "v": 8,
    }
    if timespan is not None:
        params["timespan"] = timespan
    if shiptype_code is not None:
        params["shiptype"] = shiptype_code

    payload = await core.upstream_get(f"/exportvessels/{apikey}", params)

    meta_dict = {
        "source": core.APP_NAME,
        "endpoint": "ais_vessels_in_aoi",
        "upstreamEndpoint": core.upstream_template("/exportvessels/{api_key}"),
        "variablesHash": core.vhash({**params, "aoi_id": aoi_id} if aoi_id else params),
        "fetchedAt": core.now_iso(),
        "version": core.APP_VERSION,
    }
    if aoi_id:
        aoi_feat = core.AOI.get(aoi_id)
        aoi_hash = core.sha256_hex(
            core.canonical_json({"properties": aoi_feat.properties, "geometry": aoi_feat.geometry})
        )
        meta_dict.update(
            {
                "aoiId": aoi_id,
                "aoiHash": aoi_hash,
                "bbox": [minLon, minLat, maxLon, maxLat],
                "registryHash": core.AOI._registry_hash,
                "aoiSource": core.AOI_PATH,
            }
        )

    response = {"nodes": payload, "meta": meta_dict}
    return response


@mcp.tool
async def ais_vessels_nearby(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_nm: float = 50.0,
    aoi_id: Optional[str] = None,
    timespan: Optional[int] = 60,
    shiptype: Optional[str] = None,
    msgtype: str = "simple",
) -> dict:
    """
    List vessels near a given coordinate or AOI centroid.

    Args:
        lat: Latitude (WGS84).
        lon: Longitude (WGS84).
        radius_nm: Radius in nautical miles.
        aoi_id: Optional AOI whose centroid is used as origin.
        timespan: Minutes back from now to consider.
        shiptype: Optional AIS ship type filter.
        msgtype: Detail level of AIS messages: simple | extended | full.

    Returns:
        dict: Vessel list payload and governance metadata.
    """
    apikey = core.AIS_EXPORTVESSELS_KEY
    if msgtype not in core.AOI_MSGTYPES:
        raise core.HTTPException(
            status_code=400, detail=f"Invalid msgtype. Allowed: {sorted(core.AOI_MSGTYPES)}"
        )

    # Derive lat/lon from AOI if provided
    if aoi_id:
        minLon0, minLat0, maxLon0, maxLat0 = core.AOI.bbox_for(aoi_id)
        lon = (minLon0 + maxLon0) / 2.0
        lat = (minLat0 + maxLat0) / 2.0
    if lat is None or lon is None:
        raise core.HTTPException(status_code=400, detail="Provide lat & lon or aoi_id")

    shiptype_code = core.normalize_shiptype(shiptype)

    minLon, minLat, maxLon, maxLat = core.bbox_around_point_nm(lon, lat, radius_nm)
    params = {
        "minlat": minLat,
        "minlon": minLon,
        "maxlat": maxLat,
        "maxlon": maxLon,
        "msgtype": msgtype,
        "protocol": "jsono",
        "v": 8,
    }
    if timespan is not None:
        params["timespan"] = timespan
    if shiptype_code is not None:
        params["shiptype"] = shiptype_code

    upstream = await core.upstream_get(f"/exportvessels/{apikey}", params)

    radius_km = radius_nm * 1.852
    if isinstance(upstream, list):
        filtered = []
        for rec in upstream:
            plat, plon = core.extract_lat_lon(rec)
            if plat is None or plon is None:
                continue
            distance = core.haversine_km(lat, lon, plat, plon)
            if distance <= radius_km:
                filtered.append(rec)
        nodes = filtered
    else:
        nodes = upstream

    meta_dict = {
        "source": core.APP_NAME,
        "endpoint": "ais_vessels_nearby",
        "upstreamEndpoint": core.upstream_template("/exportvessels/{api_key}"),
        "variablesHash": core.vhash(
            {**params, "lat": lat, "lon": lon, "radius_nm": radius_nm, "aoi_id": aoi_id}
        ),
        "fetchedAt": core.now_iso(),
        "version": core.APP_VERSION,
    }
    if aoi_id:
        aoi_feat = core.AOI.get(aoi_id)
        aoi_hash = core.sha256_hex(
            core.canonical_json({"properties": aoi_feat.properties, "geometry": aoi_feat.geometry})
        )
        meta_dict.update(
            {
                "aoiId": aoi_id,
                "aoiHash": aoi_hash,
                "bbox": [minLon, minLat, maxLon, maxLat],
                "registryHash": core.AOI._registry_hash,
                "aoiSource": core.AOI_PATH,
            }
        )

    response = {"nodes": nodes, "meta": meta_dict}
    return response


@mcp.tool
async def get_vessel_info(
    mmsi: Optional[str] = None,
    imo: Optional[str] = None,
    shipname: Optional[str] = None,
) -> dict:
    """
    Retrieve detailed information for a single vessel.

    Args:
        mmsi: Maritime Mobile Service Identity.
        imo: IMO vessel number.
        shipname: Optional ship name.

    Returns:
        dict: Vessel info nodes and governance metadata.
    """

    payload = None
    if imo:
        payload = core.fetch_vessel_info_by_imo(imo, after_cursor=None)
    if mmsi:
        payload = core.fetch_vessel_info_by_mmsi(mmsi, after_cursor=None)
    if shipname:
        payload = core.fetch_vessel_info_by_name(shipname, after_cursor=None)

    if payload is None:
        raise core.HTTPException(
            status_code=400,
            detail="Provide at least one of: mmsi, imo, or shipname.",
        )

    meta = core.GovernanceMeta(
        source=core.APP_NAME,
        endpoint="get_vessel_info",
        fetchedAt=core.now_iso(),
        version=core.APP_VERSION,
    )
    nodes = payload["data"]["vessels"]["nodes"]
    response = {"nodes": nodes, "meta": meta.model_dump()}
    return response


def run() -> None:
    """
    Entry point for running the AIS FastMCP server process.

    Reads HOST, PORT, and MCP_TRANSPORT from the environment and starts the
    Streamable HTTP server on the configured address.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8200"))
    transport = os.getenv("MCP_TRANSPORT", "http")
    mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run()


