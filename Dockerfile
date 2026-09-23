# Model Passport: CLI, registry API, and dashboard in one image.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY dashboard ./dashboard
RUN pip install ".[registry,dashboard]"

# Run as an unprivileged user; the registry database lives on a volume.
RUN useradd --create-home --uid 10001 passport && mkdir -p /data && chown passport /data
USER passport
ENV PASSPORT_REGISTRY_DB=/data/registry.db

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["passport", "serve", "--host", "0.0.0.0", "--port", "8000", "--db", "/data/registry.db"]
