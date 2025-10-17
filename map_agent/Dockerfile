
# Minimal Dockerfile for the Arctic Map Agent
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY templates ./templates
COPY static ./static
COPY map_openapi.yaml map_openapi.yaml

# Default environment
ENV OPENAPI_FILE=maps_openapi.yml
ENV HOST=0.0.0.0
ENV PORT=8120

# Set START_CMD to your library's launcher if different.
# Example: START_CMD="ais-agent --openapi ${OPENAPI_FILE} --host ${HOST} --port ${PORT}"
ENV START_CMD="uvicorn map_agent:app --host ${HOST} --port ${PORT} --reload"

# Expose default port
EXPOSE 8120

# Optional healthcheck; disable or adjust path if your agent differs
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/map/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
