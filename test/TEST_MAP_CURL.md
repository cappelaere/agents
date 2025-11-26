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


curl -X POST $BASE/ingest \
  -H 'Content-Type: application/json' \
  -d '
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T00:15:19",
        "speed_knots": 6,
        "course_deg": 13,
        "heading_deg": 334,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35672, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T01:00:22",
        "speed_knots": 6,
        "course_deg": 22,
        "heading_deg": 311,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35797, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T01:45:24",
        "speed_knots": 6,
        "course_deg": 4,
        "heading_deg": 312,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T02:06:26",
        "speed_knots": 6,
        "course_deg": 4,
        "heading_deg": 312,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T02:27:25",
        "speed_knots": 6,
        "course_deg": 4,
        "heading_deg": 312,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T02:48:24",
        "speed_knots": 6,
        "course_deg": 20,
        "heading_deg": 312,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T03:09:27",
        "speed_knots": 6,
        "course_deg": 71,
        "heading_deg": 311,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T03:30:28",
        "speed_knots": 6,
        "course_deg": 138,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T03:51:27",
        "speed_knots": 6,
        "course_deg": 142,
        "heading_deg": 311,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T04:12:24",
        "speed_knots": 6,
        "course_deg": 172,
        "heading_deg": 311,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T04:33:29",
        "speed_knots": 6,
        "course_deg": 22,
        "heading_deg": 311,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T04:54:29",
        "speed_knots": 6,
        "course_deg": 18,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T05:15:29",
        "speed_knots": 6,
        "course_deg": 27,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T05:36:27",
        "speed_knots": 6,
        "course_deg": 27,
        "heading_deg": 311,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T05:57:28",
        "speed_knots": 6,
        "course_deg": 30,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T06:18:29",
        "speed_knots": 6,
        "course_deg": 43,
        "heading_deg": 311,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T06:39:29",
        "speed_knots": 6,
        "course_deg": 45,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T07:00:32",
        "speed_knots": 6,
        "course_deg": 121,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T07:21:27",
        "speed_knots": 6,
        "course_deg": 67,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T07:42:31",
        "speed_knots": 6,
        "course_deg": 23,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T08:06:32",
        "speed_knots": 6,
        "course_deg": 97,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T08:30:35",
        "speed_knots": 6,
        "course_deg": 130,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T08:51:30",
        "speed_knots": 6,
        "course_deg": 140,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T09:12:36",
        "speed_knots": 6,
        "course_deg": 116,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T09:33:32",
        "speed_knots": 6,
        "course_deg": 81,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T09:54:35",
        "speed_knots": 6,
        "course_deg": 23,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T10:15:28",
        "speed_knots": 6,
        "course_deg": 34,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T10:36:29",
        "speed_knots": 6,
        "course_deg": 159,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T10:57:33",
        "speed_knots": 6,
        "course_deg": 145,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T11:18:28",
        "speed_knots": 6,
        "course_deg": 138,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T11:39:34",
        "speed_knots": 6,
        "course_deg": 164,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T12:00:31",
        "speed_knots": 6,
        "course_deg": 115,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T12:21:35",
        "speed_knots": 6,
        "course_deg": 151,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T12:45:29",
        "speed_knots": 6,
        "course_deg": 173,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T13:06:30",
        "speed_knots": 6,
        "course_deg": 157,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T13:27:35",
        "speed_knots": 6,
        "course_deg": 148,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T13:48:30",
        "speed_knots": 6,
        "course_deg": 58,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T14:09:30",
        "speed_knots": 6,
        "course_deg": 62,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T14:30:36",
        "speed_knots": 6,
        "course_deg": 54,
        "heading_deg": 310,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    },
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-11-24T14:51:34",
        "speed_knots": 6,
        "course_deg": 57,
        "heading_deg": 309,
        "status": "0"
      },
      "geometry": { "type": "Point", "coordinates": [-146.35786, 61.122799] }
    }
  ]
}'