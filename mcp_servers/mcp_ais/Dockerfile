FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

# Ensure entrypoint is executable (COPY . can reset permissions)
RUN chmod +x /app/docker-entrypoint.sh

# Default environment
ENV HOST=0.0.0.0
ENV PORT=8200
ENV LOG_LEVEL=info

# Start the FastMCP-based AIS MCP server by default.
# The code is copied into /app and treated as a flat module.
ENV START_CMD="python -m fastmcp_app"

EXPOSE $PORT
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
ENTRYPOINT ["./docker-entrypoint.sh"]
