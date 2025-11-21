from langfuse import Langfuse
import os, logging

logger = logging.getLogger("langfuse_utils")
logger.setLevel(logging.DEBUG)

secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
base_url = os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "http://150.240.3.116:3000"))

if secret_key and public_key:
    langfuse = Langfuse(secret_key=secret_key, public_key=public_key, host=base_url)
else:
    langfuse = None
    logger.warning("Langfuse disabled: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set")


def get_session_id(request):
    session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("session-id")
        or request.headers.get("session_id")
        or request.headers.get("x-sessionid")
        or request.headers.get("sessionid")
    )    
    return session_id
    
def trace_start(request):
    session_id = get_session_id(request)
    query_inputs = dict(request.query_params)
    inputs = {
        "session_id": session_id,
        "query": query_inputs,
        "method": request.method,
        "path": request.url.path,
    }
    if langfuse is None:
        logger.debug(f"Langfuse disabled, trace_start no-op for {inputs}")
        return None
    trace = langfuse.start_span(name=request.url.path)
    trace.update(input=inputs)
    logger.info(inputs)
    return trace

def trace_end(trace, resp):
    if trace is None:
        return
    try:
        trace.update(output=resp)
        trace.end()
    except Exception:
        logger.exception("langfuse flush failed")


def trace_flush():
    if langfuse is None:
        return
    langfuse.flush()
