# ...existing code...
import os
from langfuse import observe, Langfuse

secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
base_url = os.getenv("LANGFUSE_BASE_URL", "http://150.240.3.116:3000")

if not secret_key or not public_key:
    raise SystemExit("LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY not set")

langfuse = Langfuse(
    secret_key=secret_key,
    public_key=public_key,
    host=base_url
)

@observe()
def my_function(input_text: str) -> str:
    return f"Processed: {input_text}"

result = my_function("Hello, Langfuse!")
print("Result:", result)

langfuse.flush()
print("Langfuse test complete")
# ...existing code...