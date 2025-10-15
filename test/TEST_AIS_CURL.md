BASE=http://localhost:8100/ais

# Heahth
curl -s "$BASE/health" | jq .

# List AOIs
curl -s "$BASE/aoi" | jq .

# Inspect one AOI
curl -s "$BASE/aoi/chukchi_sea_ak" | jq .

# AOI — cargo, 2h, extended detail
curl -s "$BASE/vessels/aoi?aoi_id=chukchi_sea_ak&timespan=1440" | jq .
curl -s "$BASE/vessels/aoi?bbox=-175,65,-155,80&timespan=1440&shiptype=7&msgtype=extended" | jq .

# Nearby — 50 nm around (82N,20E), tankers, full detail
curl -s "$BASE/vessels/nearby?lat=65.1&lon=-170.7&radius_nm=100&msgtype=simple" | jq .

# Vessel photo — by ship_id
curl -s "$BASE/vessel/photo?ship_id=689883" | jq .

# Vessel info — by MMSI
curl -s "$BASE/vessel/info?mmsi=257017000" | jq .
curl -s "$BASE/vessel/info?imo=273214780" | jq .
curl -s "$BASE/vessel/info?shipname=BOSS" | jq .

# Vessel track — by IMO within a time window
curl -s "$BASE/vessel/track?imo=9538907&fromdt=2025-09-01%2000:00&todt=2025-09-02%2000:00" | jq .

# Vessel track — by ship_id with a timespan (last 12 hours)
curl -s "$BASE/vessel/track?ship_id=1234567&timespan=720" | jq .

