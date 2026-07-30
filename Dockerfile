# ---- Stage 1: builder -------------------------------------------------------
FROM continuumio/miniconda3:24.1.2-0 AS builder

WORKDIR /build

# System deps for compiled wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create the conda environment with RDKit from conda-forge
RUN conda create -n molops -c conda-forge python=3.11 rdkit -y

# Make conda env the active Python
ENV PATH="/opt/conda/envs/molops/bin:$PATH"

# Install remaining pip dependencies into the conda env
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev]"

# ---- Stage 2: runtime -------------------------------------------------------
FROM debian:bookworm-slim AS runtime

ARG BUILD_DATE
ARG GIT_SHA
ARG VERSION=latest

LABEL org.opencontainers.image.title="MolOps API"
LABEL org.opencontainers.image.description="MLOps-driven cheminformatics pipeline for drug bioactivity prediction"
LABEL org.opencontainers.image.source="https://github.com/LYHAMSEA/MolOps"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.version="${VERSION}"

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 molops && \
    useradd --uid 1001 --gid molops --shell /bin/bash --create-home molops

WORKDIR /app

# Copy the entire conda environment from builder
COPY --from=builder /opt/conda/envs/molops /opt/conda/envs/molops
ENV PATH="/opt/conda/envs/molops/bin:$PATH"

# Copy application source
COPY --chown=molops:molops molops/ ./molops/
COPY --chown=molops:molops pyproject.toml ./

# Copy trained models (if they exist -- otherwise API starts in degraded mode)
COPY --chown=molops:molops models/ ./models/

RUN pip install --no-deps -e .

USER molops

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

CMD ["uvicorn", "molops.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]