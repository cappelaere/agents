BASE=http://localhost:8100


# List AOIs
curl -s "$BASE/aoi" | jq .

# Inspect one AOI
curl -s "$BASE/aoi/chukchi_sea_ak" | jq .

# 1) Search by name
curl -s "http://localhost:8100/vessels/search?shipname=POLAR" | jq .

# 2) AOI — cargo, 2h, extended detail
curl -s "$BASE/vessels/aoi?aoi_id=chukchi_sea_ak&timespan=1440" | jq .
curl -s "$BASE/vessels/aoi?bbox=-175,65,-155,80&timespan=1440&shiptype=7&msgtype=extended" | jq .

# 3) Nearby — 50 nm around (82N,20E), tankers, full detail
curl -s "$BASE/vessels/nearby?lat=65.1&lon=-170.7&radius_nm=100&msgtype=simple" | jq .

# 4) Vessel photo — by ship_id
curl -s "$BASE/vessel/photo?ship_id=360871" | jq .

# 4) Search Vessel — by ship_id
curl -s "$BASE/vessels/search?mmsi=273214780" | jq .
curl -s "$BASE/vessels/search?ship_id=675872" | jq .
curl -s "$BASE/vessels/search?imo=9411410" | jq .
curl -s "$BASE/vessels/search?shipname=BOSS" | jq .

# 5) Vessel info — by MMSI
curl -s "http://localhost:8100/vessel/info?mmsi=257017000" | jq .

curl -s "http://localhost:8100/vessel/info?imo=9411410" | jq .
curl -s "http://localhost:8100/vessel/info?imo=273214780" | jq .
curl -s "http://localhost:8100/vessel/info?shipname=BOSS" | jq .

# 6) Vessel track — by IMO within a time window
curl -s "http://localhost:8100/vessel/track?imo=9538907&fromdt=2025-09-01%2000:00&todt=2025-09-02%2000:00" | jq .

# 7) Vessel track — by ship_id with a timespan (last 12 hours)
curl -s "http://localhost:8100/vessel/track?ship_id=1234567&timespan=720" | jq .

