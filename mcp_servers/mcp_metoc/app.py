"""
FastMCP server exposing Open-Meteo (METOC) tools as MCP tools.

Author: Patrice G. Cappelaere, IBM Federal

Run locally with:
    python -m mcp_metoc.app
"""
from __future__ import annotations

import os
from typing import Optional

from fastmcp import FastMCP  # type: ignore

from . import metoc_client

mcp = FastMCP(name="Open METOC Server")

@mcp.tool
async def get_metoc_health() -> dict:
    """Simple health probe."""
   
    response = {"status": "ok", "provider": "open-meteo"}   
    return response

@mcp.tool
async def search_metoc_geocode(name: str, count: int = 5, language: Optional[str] = None) -> dict:
    """
    Search Open-Meteo's geocoding API for matching locations.

    Args:
        name: Free-text location name (city, region, etc.).
        count: Maximum number of results to return.
        language: Optional language code for localized place names.

    Returns:
        dict: Dictionary containing the original query and Open-Meteo geocoding results.
    """
    try:
        results = await metoc_client.geocode_search(name=name, count=count, language=language, fmt="json")
        response = {"query": {"name": name, "count": count, "language": language}, "results": results}
        return response
    except Exception as e:
        raise e

@mcp.tool
async def get_atmosphere_forecast(
    lat: float,
    lon: float,
    hourly: Optional[str] = None,
    daily: Optional[str] = None,
    current_weather: bool = True,
    timezone: Optional[str] = None,
    forecast_days: int = 7,
) -> dict:
    """Fetch atmospheric forecasts from Open-Meteo."""
   
    try:
        forecast = await metoc_client.atmosphere_forecast(
            lat=lat,
            lon=lon,
            hourly=hourly,
            daily=daily,
            current_weather=current_weather,
            timezone=timezone,
            forecast_days=forecast_days,
        )
        response = {"query": {"lat": lat, "lon": lon}, "forecast": forecast}
        return response
    except Exception as e:
        raise e

# ------------------ Archive -----------------------------
@mcp.tool
async def get_atmosphere_archive( 
    lat: float, 
    lon: float, 
    start_date: str, 
    end_date: str, 
    hourly: Optional[str] = None, 
    daily: Optional[str] = None, 
    timezone: Optional[str] = None) -> dict:
    """
    Fetch historical atmospheric data via the Open-Meteo archive API.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        hourly: Optional comma-separated list of hourly variables.
        daily: Optional comma-separated list of daily variables.
        timezone: Optional IANA timezone name.

    Returns:
        dict: Dictionary with the original query and archive payload from Open-Meteo.
    """

    try:
        archive = await metoc_client.atmosphere_archive(lat=lat, lon=lon, start_date=start_date, end_date=end_date, hourly=hourly, daily=daily, timezone=timezone)
        response = {"query": {"lat": lat, "lon": lon, "start_date": start_date, "end_date": end_date}, "archive": archive}
        return response
    except Exception as e:
        raise e

@mcp.tool
async def get_marine_forecast(
    lat: float,
    lon: float,
    hourly: Optional[str] = None,
    timezone: Optional[str] = None,
    forecast_days: int = 5,
) -> dict:
    """
    Fetch marine forecast variables (wave height, currents, etc.) from Open-Meteo.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        hourly: Optional comma-separated list of hourly marine variables.
        timezone: Optional IANA timezone name.
        forecast_days: Number of days ahead to forecast.

    Returns:
        dict: Dictionary with the original query and marine forecast payload.
    """
    try:
        marine = await metoc_client.marine_forecast(lat=lat, lon=lon, hourly=hourly, timezone=timezone, forecast_days=forecast_days)
        response = {"query": {"lat": lat, "lon": lon}, "marine": marine}
        return response
    except Exception as e:
        raise e


def run() -> None:
    """
    Entry point for running the FastMCP server process.

    Reads HOST, PORT, and MCP_TRANSPORT from the environment and starts the
    Streamable HTTP server on the configured address.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8201"))
    transport = os.getenv("MCP_TRANSPORT", "http")
    mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run()

