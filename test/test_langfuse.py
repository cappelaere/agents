import os
import json
import time
import requests

# Read from your .env (already set in /home/vpcuser/agents/.env.sh)
BASE_URL = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000").rstrip("/")
API_KEY = os.getenv("LANGFUSE_SECRET_KEY")
if not API_KEY:
    raise SystemExit("LANGFUSE_SECRET_KEY not set in environment")

endpoint = f"{BASE_URL}/api/events"  # adjust path if your Langfuse instance uses a different ingest path

now = int(time.time() * 1000)
payload = {
    "type": "run",                    # generic type — adapt to your schema
    "project": "test-project",
    "run": {
        "id": f"py-test-{now}",
        "name": "python-test-run",
        "status": "success",
        "start_time": now - 1000,
        "end_time": now
    },
    "inputs": {"prompt": "Say hello"},
    "outputs": {"text": "Hello from Langfuse test"},
    "metadata": {"env": "dev", "note": "quick smoke test"}
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=10)
print("POST", endpoint)
print("Status:", resp.status_code)
try:
    print("Response JSON:", resp.json())
except Exception:
    print("Response text:", resp.text)