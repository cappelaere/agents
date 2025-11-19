# Testing Langfuse

```
source .env.sh
curl -v -H "Authorization: Bearer $LANGFUSE_SECRET_KEY" -H "Content-Type: application/json" \
  -d '{"test":"ping"}' \
  "http://150.240.3.116:3000/api/v1/ingest"
```