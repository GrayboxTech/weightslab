# =============================================================================
# WeightsLab "training docker" — Docker-in-Docker (DinD) variant
# =============================================================================
# This image runs BOTH:
#   1. the WeightsLab CLI ("weightslab ui launch") which spins up the
#      Envoy + Weights Studio frontend containers, and
#   2. the training process ("weightslab start example") which serves the
#      in-process gRPC backend on :50051.
#
# DinD = the container runs its OWN docker daemon inside itself (requires
# --privileged). The Envoy/frontend containers are nested *inside* this
# container's daemon, so they share this container's network namespace and
# filesystem. See README.md for why that matters.
# =============================================================================
FROM python:3.11-slim

# --- System deps -------------------------------------------------------------
# - docker engine (dockerd + CLI + compose plugin + containerd): installed via
#   the official convenience script. We need the *daemon* here (DinD).
# - curl/ca-certificates/git: fetch the docker installer + optional dev install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl sudo ca-certificates git \
    && curl -fsSL https://get.docker.com | sh \
    && rm -rf /var/lib/apt/lists/*

# --- WeightsLab --------------------------------------------------------------
# Default: install the published package from PyPI (matches "if you didn't
# modify weightslab, use pip install"). To run your local dev branch instead:
#   docker compose build \
#     --build-arg WEIGHTSLAB_SPEC="git+https://github.com/GrayboxTech/weightslab.git@dev"
ARG WEIGHTSLAB_SPEC=weightslab
RUN pip install --no-cache-dir "${WEIGHTSLAB_SPEC}"

# gRPC backend port — must match what Envoy is told to dial (GRPC_BACKEND_PORT).
ENV GRPC_BACKEND_PORT=50051

# --- GPU (NVIDIA) ------------------------------------------------------------
# Make the NVIDIA Container Toolkit inject the host driver (nvidia-smi + libs)
# into this container. `utility` => nvidia-smi works; `compute` => CUDA/torch.
# These are no-ops on a host without an NVIDIA GPU/toolkit (falls back to CPU).
# The actual GPU grant is requested in docker-compose.yml (deploy.resources).
# torch's default Linux wheel bundles the CUDA runtime, so no CUDA base image is
# needed — only the host driver (injected) is required for torch.cuda to work.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# Strip any CR so the script runs under Linux bash even if checked out on Windows.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# OPTIONAL - Documentation only — EXPOSE does NOT publish ports. The host publishing is done
# by `ports:` in docker-compose.yml. Listed here purely to record intent:
#   5173 = Weights Studio frontend, 8080 = Envoy gRPC-web.
EXPOSE 5173 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
