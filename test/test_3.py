import os
import uuid
from langfuse import observe, Langfuse

secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
base_url = os.getenv("LANGFUSE_BASE_URL", "http://150.240.3.116:3000")

if not secret_key or not public_key:
    raise SystemExit("LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY not set")

client = Langfuse(secret_key=secret_key, public_key=public_key, host=base_url)

uid = str(uuid.uuid4())

@observe()
def make_trace(text: str) -> str:
    # include the UUID in the output so it appears in logs/UI
    return f"Processed: {text} -- id:{uid}"

print("sending trace id:", uid)
res = make_trace("smoke-test")
client.flush()
print("flushed; search for id in logs/UI:", uid)