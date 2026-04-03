FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TZ=America/New_York

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY src ./src
COPY docker/run_serious_batch.sh /usr/local/bin/run-serious-batch
COPY docker/run_live_strategy.sh /usr/local/bin/run-live-strategy
COPY README.md ./

RUN chmod +x /usr/local/bin/run-serious-batch /usr/local/bin/run-live-strategy && \
    mkdir -p /app/data /app/reports

ENTRYPOINT ["/usr/local/bin/run-serious-batch"]
