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

# Set START_CMD to your library's launcher if different.
# Example: START_CMD="ais-agent --openapi ${OPENAPI_FILE} --host ${HOST} --port ${PORT}"
ENV START_CMD="uvicorn app:app --host ${HOST} --port ${PORT} --log-level $LOG_LEVEL --reload"

EXPOSE $PORT
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
ENTRYPOINT ["./docker-entrypoint.sh"]
