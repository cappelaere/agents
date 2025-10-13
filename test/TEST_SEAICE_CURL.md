# Testing SEAICE Agent with Curl commands

```
uvicorn seaice_agent:app --port 8090
```
bash
BASE=http://localhost:8090
RID=$(uuidgen)   # or any string; optional


1) Health
curl -i "$BASE/seaice/health" \
  -H "X-Request-ID: ${RID:-test-health-001}"

2) WMS template
curl -i "$BASE/seaice/wms?layer=seaice_conc&time=2025-09-01&bbox=60,-180,90,180&srs=EPSG:4326&width=1024&height=512" \
  -H "X-Request-ID: ${RID:-test-wms-001}"

3) Ensure/download a file (GET)
# cached-or-download
curl -i "$BASE/seaice/download?time=2025-09-01" \
  -H "X-Request-ID: ${RID:-test-dl-001}"

# force re-download
curl -i "$BASE/seaice/download?time=2025-09-01&force=true" \
  -H "X-Request-ID: ${RID:-test-dl-002}"

4) Point sample (GET)
# lat/lon near Arctic ocean; adjust as needed
curl -i "$BASE/seaice/point?lat=82.0&lon=20.0&time=2025-09-01" \
  -H "X-Request-ID: ${RID:-test-point-001}"

5) Stats over a bbox (POST)
curl -i "$BASE/seaice/stats" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: ${RID:-test-stats-001}" \
  -d '{
        "bbox": [70, -40, 85, 40],
        "time": "2025-09-01"
      }'