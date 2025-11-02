BASE=http://$HOST:8120

# Health
curl -s "$BASE/health" | jq .

curl -s "$BASE/version" | jq .

# to test the ui
open browser: http://$BASE

# and issue curl commands to display result son the map

curl -X POST $BASE/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "type":"FeatureCollection",
    "features":[
      {"type":"Feature","geometry":{"type":"Point","coordinates":[-150,75]},"properties":{"name":"Vessel A","mmsi":"123"}},
      {"type":"Feature","geometry":{"type":"Point","coordinates":[-160,78]},"properties":{"name":"Vessel B","mmsi":"456"}},
      {"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-170,72],[-160,72],[-160,74],[-170,74],[-170,72]]]},"properties":{"name":"AOI"}}
    ]
  }'

  # Send a local GeoJSON file
curl -X POST $BASE/ingest \
  -H 'Content-Type: application/json' \
  --data-binary @aois.geojson

