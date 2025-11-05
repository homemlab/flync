# FLYNC Multi-Stage Dockerfile
#
# This Dockerfile provides two build targets:
#   1. flync-runtime (default): Full environment with runtime track downloading
#   2. flync-prewarmed: Pre-downloads tracks during build for faster startup
#
# Build examples:
#   docker build -t flync:latest .
#   docker build -t flync:prewarmed --target flync-prewarmed --build-arg BWQ_CONFIG=config/bwq_config.yaml .

# =============================================================================
# Stage 1: Base environment with Conda and dependencies
# =============================================================================
FROM continuumio/miniconda3:latest AS base

# Metadata
LABEL maintainer="FLYNC Contributors"
LABEL description="FLYNC - lncRNA discovery pipeline for Drosophila melanogaster"
LABEL version="1.0.0"

# Set environment
ENV TERM=xterm-256color
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        git \
        build-essential \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/flync

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -a -y

# Initialize conda for bash
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate flync" >> ~/.bashrc

# =============================================================================
# Stage 2: Install FLYNC package (runtime target)
# =============================================================================
FROM base AS flync-runtime

# Copy source code
COPY . .

# Install FLYNC package in development mode
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate flync && \
    pip install -e .

# Create cache directory for BigWig tracks (will be populated at runtime)
RUN mkdir -p /opt/flync/bwq_persistent_cache

# Set environment for track caching
ENV FLYNC_CACHE_DIR=/opt/flync/bwq_persistent_cache

# Expose volume for data and results
VOLUME ["/data", "/results"]

# Set entrypoint to activate conda environment
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate flync && exec \"$@\"", "--"]

# Default command: show help
CMD ["flync", "--help"]

# =============================================================================
# Stage 3: Pre-warmed image with cached tracks (prewarmed target)
# =============================================================================
FROM flync-runtime AS flync-prewarmed

# Build argument for BWQ config file
ARG BWQ_CONFIG=src/flync/config/bwq_config.yaml

# Copy BWQ config if provided
COPY ${BWQ_CONFIG} /opt/flync/bwq_config_build.yaml

# Pre-download all tracks specified in config
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate flync && \
    echo "Pre-downloading genomic tracks..." && \
    python src/flync/workflows/scripts/predownload_tracks.py \
        /opt/flync/bwq_config_build.yaml \
        /opt/flync/bwq_persistent_cache && \
    echo "Tracks pre-downloaded successfully!" && \
    rm /opt/flync/bwq_config_build.yaml

# Metadata for prewarmed image
LABEL variant="prewarmed"
LABEL description="FLYNC with pre-cached genomic tracks for faster startup"

# Default command: show help
CMD ["flync", "--help"]
