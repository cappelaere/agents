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

curl -s "$BASE/vessel/track?ship_id=307571&protocol=jsono&msgtype=simple&v=3&fromdate=2025-09-01&todate=2025-10-10" | jq .

https://services.marinetraffic.com/api/exportvesseltrack/c49b0d8a02dc441b8a75b7a3bf32d216fdd13032?shipid=307571&protocol=jsono&msgtype=simple&v=3&fromdate=2025-09-01&todate=2025-10-10
https://services.marinetraffic.com/api/exportvesseltrack/fd74c58e32b15115a3d78b467cc6877c8f9746b2?shipid=307571&protocol=jsono&v=3&msgtype=simple 

# Vessel track — by ship_id with a timespan (last 12 hours)
curl -s "$BASE/vessel/track?ship_id=1234567&timespan=720" | jq .

# Vessel portcalls by ship id
curl -s "$BASE/vessel/portcalls?ship_id=1234567&timespan=720" | jq .

curl -s "https://services.marinetraffic.com/api/portcalls/3a06b272b24a976ee9bd4c874443572a4c4f2b3e?shipid=675872timespan%3D720&protocol=jsono&v=6&msgtype=simple" | jq .

# Vessel events — by ship_id with a timespan (last 12 hours)
curl -s "$BASE/vessel/events?mmsi=338718000" | jq .

# Routing distance from port to port
curl -s "$BASE/routing/distance_to_port?start_port=JPTOK&end_port=USHOU" | jq .

curl "https://services.marinetraffic.com/api/vesselevents/d5794deb0dd5cc5364def3afcc905373b22c494b?mmsi=338718000&protocol=jsono&v=2"

curl "https://services.marinetraffic.com/api/portcalls/3a06b272b24a976ee9bd4c874443572a4c4f2b3e?timespan=2&msgtype=simple&portid=NLAMS&protocol=jsono&v=6" | jq


curl -s "$BASE/portcalls?port_id=USANC&timespan=2880" | jq .
curl "https://services.marinetraffic.com/api/portcalls/3a06b272b24a976ee9bd4c874443572a4c4f2b3e?portid=USANC&protocol=jsono&v=6&msgtype=simple&timespan=2880" | jq

https://services.marinetraffic.com/api/portcalls/3a06b272b24a976ee9bd4c874443572a4c4f2b3e?portid=USANC&protocoll=jsono&v=6&msgtype=simple&timespan=2880

# Check ships in Chukchi Sea
curl -s "$BASE/vessels/aoi?aoi_id=chukchi_sea_ak&timespan=1440" | jq .

curl -s "$BASE/routing/distance_to_port?start_port=USACP&end_port=USANC" | jq .

# Distance of one ship to Wainwright AK (USAIN) to Prudhoe Bay (USSCC)
curl -s "$BASE/routing/distance_to_port?start_port=USANC&end_port=USSCC" | jq .

curl -s "https://services.marinetraffic.com/api/exportroutes/3a1ad4951588c7b215acd23029fea867ea97b9c5?port_target_id=USANC&protocol=jsono&msgtype=extended&shipid=304642"