# =============================================================================
# WeightsLab "training docker" — self-contained siblings (no host bind mounts)
# =============================================================================
# Like the siblings variant, this drives the HOST docker daemon (mounted socket)
# so Envoy + frontend run as siblings. BUT it removes the host bind-mount
# dependency entirely:
#   - Envoy's config is delivered through a named VOLUME, populated over the
#     docker socket (no host path needed).
#   - the frontend runs in HTTP mode, which needs NO certs mount at all.
# Consequence: NO path alignment, NO setup-host.sh, and it runs natively on
# Windows / Docker Desktop. The trade-off: it does NOT call `weightslab ui
# launch` — it brings the UI up from its own bind-mount-free compose
# (ui-compose.yml), reusing the same Envoy + graybx/weightslab images.
# =============================================================================
FROM python:3.11-slim

# --- System deps -------------------------------------------------------------
# docker CLI + compose plugin ONLY (no daemon — we use the host's daemon).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl sudo ca-certificates gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# --- WeightsLab --------------------------------------------------------------
# Plain pip install — NO editable/path-aligned install needed here, because the
# host daemon never has to resolve a package path (no host bind mounts). To run
# the dev branch instead:
#   --build-arg WEIGHTSLAB_SPEC="git+https://github.com/GrayboxTech/weightslab.git@dev"
ARG WEIGHTSLAB_SPEC=weightslab
RUN pip install --no-cache-dir "${WEIGHTSLAB_SPEC}"

ENV GRPC_BACKEND_PORT=50051

# --- GPU (NVIDIA) ------------------------------------------------------------
# Make the NVIDIA Container Toolkit inject the host driver (nvidia-smi + libs).
# No-op without an NVIDIA GPU/toolkit; the actual grant is in docker-compose.yml.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# The bind-mount-free UI compose, read by the docker CLI inside this container
# and sent to the host daemon. It references only images + a named volume.
COPY ui-compose.yml /opt/ui/ui-compose.yml
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
