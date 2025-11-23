"""
Shared Open-Meteo HTTP helpers used by both REST and MCP servers.

Author: Patrice G. Cappelaere, IBM Federal
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ATMOSPHERE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

OPEN_METEO_TIMEOUT = float(os.getenv("OPEN_METEO_TIMEOUT", "20"))
USER_AGENT = os.getenv("OPEN_METEO_USER_AGENT", "mcp-metoc/0.1 (+https://modelcontextprotocol.io)")

HEADERS = {"User-Agent": USER_AGENT}


async def _request(url: str, params: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Perform a single GET request against an Open-Meteo endpoint.

    Args:
        url: Base URL of the Open-Meteo API.
        params: Query parameters (values may be None and are filtered out).

    Returns:
        dict: Parsed JSON response.

    Raises:
        httpx.HTTPError: If the upstream request fails.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(OPEN_METEO_TIMEOUT), headers=HEADERS) as client:
        response = await client.get(url, params={k: v for k, v in params.items() if v is not None})
        response.raise_for_status()
        return response.json()


async def geocode_search(name: str, count: int, language: Optional[str], fmt: str = "json") -> Dict[str, Any]:
    """
    Call the Open-Meteo geocoding API.

    Args:
        name: Free-text location name.
        count: Maximum number of results to return.
        language: Optional language code for localized names.
        fmt: Response format (usually "json").

    Returns:
        dict: Raw JSON response from the geocoding endpoint.
    """
    return await _request(
        GEOCODE_URL,
        {"name": name, "count": str(count), "language": language, "format": fmt},
    )


async def atmosphere_forecast(
    lat: float,
    lon: float,
    hourly: Optional[str],
    daily: Optional[str],
    current_weather: bool,
    timezone: Optional[str],
    forecast_days: int,
) -> Dict[str, Any]:
    """
    Call the Open-Meteo atmosphere forecast API.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        hourly: Optional comma-separated list of hourly variables.
        daily: Optional comma-separated list of daily variables.
        current_weather: Whether to include current conditions.
        timezone: Optional IANA timezone name.
        forecast_days: Number of days ahead to forecast.

    Returns:
        dict: Raw JSON response from the forecast endpoint.
    """
    return await _request(
        ATMOSPHERE_URL,
        {
            "latitude": f"{lat}",
            "longitude": f"{lon}",
            "hourly": hourly,
            "daily": daily,
            "current_weather": str(current_weather).lower(),
            "timezone": timezone,
            "forecast_days": str(forecast_days),
        },
    )


async def atmosphere_archive(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    hourly: Optional[str],
    daily: Optional[str],
    timezone: Optional[str],
) -> Dict[str, Any]:
    """
    Call the Open-Meteo atmosphere archive API.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        hourly: Optional comma-separated list of hourly variables.
        daily: Optional comma-separated list of daily variables.
        timezone: Optional IANA timezone name.

    Returns:
        dict: Raw JSON response from the archive endpoint.
    """
    return await _request(
        ARCHIVE_URL,
        {
            "latitude": f"{lat}",
            "longitude": f"{lon}",
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly,
            "daily": daily,
            "timezone": timezone,
        },
    )


async def marine_forecast(
    lat: float,
    lon: float,
    hourly: Optional[str],
    timezone: Optional[str],
    forecast_days: int,
) -> Dict[str, Any]:
    """
    Call the Open-Meteo marine forecast API.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        hourly: Optional comma-separated list of hourly marine variables.
        timezone: Optional IANA timezone name.
        forecast_days: Number of days ahead to forecast.

    Returns:
        dict: Raw JSON response from the marine endpoint.
    """
    return await _request(
        MARINE_URL,
        {
            "latitude": f"{lat}",
            "longitude": f"{lon}",
            "hourly": hourly,
            "timezone": timezone,
            "forecast_days": str(forecast_days),
        },
    )

