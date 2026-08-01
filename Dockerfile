# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Source must be present before `pip install .` — setuptools builds this
# project's own package (see [tool.setuptools.packages.find] in pyproject.toml),
# not just its dependencies, so a bare `COPY pyproject.toml .` isn't enough.
COPY pyproject.toml .
COPY agents/       ./agents/
COPY api/          ./api/
COPY rag/          ./rag/
COPY config/       ./config/
COPY monitoring/   ./monitoring/

# .[offline] pulls in sentence-transformers so OFFLINE_MODE works with zero
# AWS credentials — drop the extra for a lean prod image once real Bedrock
# credentials are configured.
RUN pip install --upgrade pip \
    && pip install --prefix=/install ".[offline]" --no-cache-dir

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="incident-remediation-ai"
LABEL org.opencontainers.image.description="Multi-Agent GenAI for Incident Remediation"

WORKDIR /app

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY agents/       ./agents/
COPY api/          ./api/
COPY rag/          ./rag/
COPY config/       ./config/
COPY monitoring/   ./monitoring/

# Data directory (FAISS index persisted via volume mount in production)
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--log-level", "info"]
