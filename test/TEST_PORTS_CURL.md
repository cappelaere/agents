BASE="http://localhost:8110/ports"

curl -s "$BASE/health" | jq

curl -s "$BASE/search?name=Port%20of%20Los%20Angeles" | jq

curl -s "$BASE/columns" | jq
curl -s "$BASE/search?name=los%20angeles&min_score=0&limit=10" | jq
curl -s "$BASE/search?name=Port%20of%20Los%20Angeles&country=US&exact=true" | jq


# 1) Simple name-only (loose)
curl -s "$BASE/search?name=Al%20Azaiba&min_score=0&limit=5" | jq

# 2) Name + country filter
curl -s "$BASE/search?name=Al%20Mazunah&country=OM&min_score=0&limit=5" | jq

# 3) Exact match (only when you know the exact spelling in your CSV)
curl -s "$BASE/search?name=Al%20Mudayq&country=OM&exact=true" | jq

# 4) Broader name with more results
curl -s "$BASE/search?name=Al&country=OM&min_score=0&limit=10" | jq