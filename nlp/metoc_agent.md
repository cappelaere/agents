Here are 10 Orchestrate-friendly queries (natural utterances) paired with the exact tool calls your **Arctic METOC Agent** exposes (via `operationId`). I’m using parameters and the `x-orchestrate` hints straight from your OpenAPI so the assistant can slot-fill reliably. 


2.

**Utterance:** “Find the coordinates for Utqiagvik.”
**Tool call:** `geocodeSearch(name="Utqiagvik", count=5, language="en")`
**Notes:** Output shape projects `latitude`, `longitude`, `display_name`; good follow-ons: `getAtmosphereForecast`, `getMarineForecast`.

3.

**Utterance:** “What will the temperature be in Utqiagvik tomorrow?”
**Tool call:** `getAtmosphereForecast(lat=71.2906, lon=-156.7886, hourly="temperature_2m", forecast_days=2, timezone="UTC", current_weather=true)`
**Notes:** Use lat/lon from prior geocode; output shape includes `start`, `end`, `temps`.

4.

**Utterance:** “Give me the next 48 hours of wind at 71.29, -156.79.”
**Tool call:** `getAtmosphereForecast(lat=71.29, lon=-156.79, hourly="wind_speed_10m,wind_direction_10m", forecast_days=2, timezone="UTC")`
**Notes:** Mirrors the sample intent; returns hourly arrays covering ~48h.

5.

**Utterance:** “What were the max temps last week at 71.29, -156.79?”
**Tool call:** `getAtmosphereArchive(lat=71.29, lon=-156.79, start_date="2025-10-27", end_date="2025-11-03", daily="temperature_2m_max", timezone="UTC")`
**Notes:** Historical daily max temps for the last 7 days.

6.

**Utterance:** “Show wave height near Barrow for the next 24 hours.”
**Tool call:** `getMarineForecast(lat=71.29, lon=-156.79, hourly="significant_wave_height,wind_wave_height", forecast_days=1, timezone="UTC")`
**Notes:** Output shape projects `times` and `swh` (significant wave height).

7.

**Utterance:** “Get current weather and winds for 64.5, -165.4 in Alaska.”
**Tool call:** `getAtmosphereForecast(lat=64.5, lon=-165.4, hourly="temperature_2m,wind_speed_10m,wind_direction_10m", forecast_days=1, timezone="UTC", current_weather=true)`
**Notes:** One-day snapshot with current conditions.

8.

**Utterance:** “Find ‘Prudhoe Bay’ and then fetch a 5-day marine forecast there.”
**Tool call (step 1):** `geocodeSearch(name="Prudhoe Bay", count=1, language="en")`
**Tool call (step 2):** `getMarineForecast(lat=<from step1>, lon=<from step1>, hourly="significant_wave_height,wind_wave_height,wind_wave_direction", forecast_days=5, timezone="UTC")`
**Notes:** Uses `followOns` pattern; Orchestrate can chain via slot mapping.

9.

**Utterance:** “Daily max/min temps for Nome from Sept 1 to Oct 8, 2025.”
**Tool call:** `getAtmosphereArchive(lat=64.5011, lon=-165.4064, start_date="2025-09-01", end_date="2025-10-08", daily="temperature_2m_max,temperature_2m_min", timezone="UTC")`
**Notes:** Uses Nome’s coords (could also geocode first).

10.

**Utterance:** “Return 10 best matches for ‘Barrow’ (legacy name) so I can pick one.”
**Tool call:** `geocodeSearch(name="Barrow", count=10, language="en")`
**Notes:** Helps disambiguate to Utqiagvik; projected fields include `latitude`, `longitude`, `display_name`.

**Headers you can safely add for robustness (optional on any call):**

* `Idempotency-Key: "<uuid-or-run-id>"` (safe replays across orchestrations)
* `X-API-Key: "<key-if-gatewayed>"` or `Authorization: Bearer <jwt>` (if secured) 

If you want, I can turn these into ready-to-run `curl` snippets or a WatsonX Orchestrate tool schema card set next.
