# Testing METOC Agent with Curl commands

```
uvicorn metoc_openmeteo_agent:app --port 8080
```
or 
```
docker compose up --no-deps -d metoc_agent
```
BASE=http://$HOST:8080/metoc

# 🧩 1. Health Check
Verify the API is up and reachable:
```
curl $BASE/health
```
✅ Expected:
{"status":"ok"}

# 🗺️ 2. Geocoder API
Convert place names ↔ coordinates using Open-Meteo’s geocoding service.
a) Forward Geocoding (place → lat/lon)
```
curl "$BASE/geocode/search?name=Thule&count=3"
```
✅ Returns JSON like:
{"results":[{"name":"Thule Air Base","latitude":76.5,"longitude":-68.7, ...}]}

# 🌬️ 3. Atmospheric Forecast (Open-Meteo)
Fetch current weather or forecasts for a specific Arctic coordinate.
curl "$BASE/atmosphere/forecast?lat=82.5&lon=-45&hourly=temperature_2m,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min"
✅ Expected: hourly & daily arrays (temperature, wind, etc.).

# 🕰️ 4. Historical Weather (Open-Meteo Archive)
Retrieve historical atmospheric data (use UTC dates).
```
curl "$BASE/atmosphere/archive?lat=82.5&lon=-45&start_date=2025-09-01&end_date=2025-10-08&hourly=temperature_2m,wind_speed_10m"
```

# 🌊 5. Marine Forecast (Waves & SST)
Get sea-state and sea surface temperature forecasts.
```
curl "$BASE/marine/forecast?lat=78.2&lon=-160.5&hourly=wave_height,wave_direction,wave_period,swell_wave_height,sea_surface_temperature"
```

# (Optional) shared vars

```
bash
BASE="http://$HOST:8080/metoc"
RID=$(uuidgen)   # or any string
```
1) Health

```
curl -sS "$BASE/health" \
  -H "X-Request-ID: ${RID:-metoc-health-001}" | jq
```

2) Geocoder: forward search
# Search for “Utqiagvik”
```
curl -sS "$BASE/geocode/search?name=Utqiagvik&count=5&language=en&format=json" \
  -H "X-Request-ID: ${RID:-metoc-geocode-001}" | jq
```

3) Atmosphere: forecast
# 7-day forecast with current weather and some hourly variables
```
curl -sS "$BASE/atmosphere/forecast?lat=71.29&lon=-156.76&hourly=temperature_2m,wind_speed_10m&current_weather=true&timezone=UTC&forecast_days=7" \
  -H "X-Request-ID: ${RID:-metoc-forecast-001}" | jq
```

4) Atmosphere: archive (historical)
# Historical range example
```
curl -sS "$BASE/atmosphere/archive?lat=71.29&lon=-156.76&start_date=2025-09-01&end_date=2025-09-10&hourly=temperature_2m,wind_speed_10m&timezone=UTC" -H "X-Request-ID: ${RID:-metoc-archive-001}" | jq
```
5) Marine: forecast
# Marine variables (pick those supported by Open-Meteo Marine, e.g., wave_height)
```
curl -sS "$BASE/marine/forecast?lat=72.0&lon=-150.0&hourly=wave_height,wave_direction,wave_period&timezone=UTC&forecast_days=5" -H "X-Request-ID: ${RID:-metoc-marine-001}" | jq
```