FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/mcp_metoc
ENV PYTHONPATH=/app
WORKDIR /app/mcp_metoc

RUN chmod +x docker-entrypoint.sh

ENV HOST=0.0.0.0
ENV PORT=8201
ENV LOG_LEVEL=info
ENV OPEN_METEO_TIMEOUT=20

EXPOSE ${PORT}

ENTRYPOINT ["./docker-entrypoint.sh"]

