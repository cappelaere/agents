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

# Test
curl -i -X POST $BASE/ingest \
  -H 'Content-Type: application/geo+json' \
  -d '{
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": { "type": "Point", "coordinates": [-150, 75] },
        "properties": { "name": "Test point" }
      }
    ]
  }'

# Geojson Variant
curl -X POST $BASE/ingest \
  -H 'Content-Type: application/json' \
  -d '{
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": {
          "mmsi": "367770990",
          "name": "MIN ZIDELL",
          "stroke": "#ff0000",
          "stroke-width": 2,
          "stroke-opacity": 0.8
        },
        "geometry": {
          "type": "LineString",
          "coordinates": [
            [-158.72444, 55.76582],
            [-156.25461, 56.844898],
            [-154.18759, 57.954182],
            [-151.03894, 59.004379],
            [-149.70836, 59.401508],
            [-148.80508, 59.622189],
            [-148.71877, 59.647625],
            [-147.3275, 60.776642],
            [-147.23801, 60.791161],
            [-147.20628, 60.796295],
            [-146.97449, 60.83812],
            [-146.95255, 60.850655],
            [-146.86169, 60.912525],
            [-146.68486, 60.851659],
            [-146.4176, 61.110008],
            [-146.35657, 61.123028],
            [-146.35664, 61.122635],
            [-146.35794, 61.123325],
            [-146.358,   61.123287],
            [-146.35786, 61.123413],
            [-146.35794, 61.12344],
            [-146.35789, 61.123379],
            [-146.35791, 61.123421],
            [-146.35783, 61.12336],
            [-146.35797, 61.123375],
            [-146.35793, 61.123322],
            [-146.35799, 61.123394],
            [-146.35788, 61.12336],
            [-146.35794, 61.123405],
            [-146.35793, 61.123322],
            [-146.35793, 61.123341],
            [-146.35794, 61.123322],
            [-146.35786, 61.123405],
            [-146.35794, 61.123322],
            [-146.35786, 61.123375],
            [-146.35794, 61.123322],
            [-146.35794, 61.123322],
            [-146.35794, 61.123322]
          ]
        }
      }
    ]
  }
}'
