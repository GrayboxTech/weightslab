# Training docker — Option A: Docker-in-Docker (DinD)

Run the whole WeightsLab stack from **one self-contained training container** that
runs its own Docker daemon. Inside it we run, in order:

1. `weightslab ui launch` → starts the **Envoy** + **Weights Studio frontend**
   containers (nested inside this container's daemon),
2. `weightslab start example --cls` → runs the classification training, which
   serves the in-process **gRPC backend** on `:50051`.

The end result: open `http://localhost:5173` in your browser and the UI talks to
the training process end-to-end.

## ⚙️ Required configuration — DinD training (weightslab + weights_studio)

The **must-have** bits to start a DinD training where the browser reaches the
backend. Each maps to a concrete line in the files in this folder:

| # | Requirement | Where | Why |
|---|---|---|---|
| 1 | **`privileged: true`** on the trainer | [docker-compose.yml](docker-compose.yml) | the inner `dockerd` cannot run otherwise |
| 2 | **Start the inner `dockerd` + wait for it** | [entrypoint.sh](entrypoint.sh) | nested daemon that hosts Envoy + frontend |
| 3 | **Persist `/var/lib/docker`** (named volume) | [docker-compose.yml](docker-compose.yml) | cache the Envoy + `graybx/weightslab` pulls across runs |
| 4 | **Publish `5173` + `8080`** to the host | [docker-compose.yml](docker-compose.yml) | the browser reaches the *nested* frontend + Envoy |
| 5 | **`GRPC_BACKEND_PORT=50051`** (matches Envoy) | [docker-compose.yml](docker-compose.yml) / [Dockerfile](Dockerfile) | Envoy's `grpc-backend:host-gateway` dials this port |
| 6 | **`WEIGHTSLAB_SKIP_DOCKER_OPS=1`** before `ui launch` | [entrypoint.sh](entrypoint.sh) | skip the in-container frontend rebuild; just pull the image |
| 7 | **Order: `ui launch` → `start example`** | [entrypoint.sh](entrypoint.sh) | UI stack up first; the example then serves the backend |
| — | *Not needed (unlike siblings):* publishing `:50051`, bind-mount path alignment | — | Envoy + backend share the trainer's netns + filesystem |

### GPU — to run `nvidia-smi` / train on CUDA

GPU access is wired straight into `docker compose up` (no extra files):

| Requirement | Where |
|---|---|
| Host: NVIDIA driver **+ NVIDIA Container Toolkit** (`sudo nvidia-ctk runtime configure --runtime=docker`) | host setup |
| `deploy.resources.reservations.devices` (driver `nvidia`, `capabilities: [gpu]`) | [docker-compose.yml](docker-compose.yml) |
| `NVIDIA_VISIBLE_DEVICES=all` + `NVIDIA_DRIVER_CAPABILITIES=compute,utility` | [Dockerfile](Dockerfile) |

> The `deploy:` GPU block is a **hard requirement**: on a host with no NVIDIA
> GPU/toolkit, comment it out in [docker-compose.yml](docker-compose.yml) or
> `up` will fail with *"could not select device driver nvidia"*. Only the
> **trainer** container needs the GPU — the nested Envoy/frontend don't.
> The entrypoint prints `nvidia-smi -L` at startup and degrades to CPU if no GPU
> is visible. On **Docker Desktop (Windows)** GPU passthrough goes through WSL2;
> combining it with `--privileged` DinD can be finicky — verify with
> `docker compose run --rm trainer nvidia-smi`. (A non-NVIDIA host, e.g. an AMD
> Radeon machine, will not expose `nvidia-smi` regardless of this config.)

## How the wiring works (and why DinD is simple)

```
host browser
  → localhost:5173 ── (re-published) ──► trainer container ──► inner frontend :5173
  → localhost:8080 ── (re-published) ──► trainer container ──► inner Envoy :8080
                                                                  │ grpc-backend:host-gateway
                                                                  ▼
                                              in-process gRPC backend :50051  (same container)
```

Because the inner daemon runs **inside** the trainer container:

- **Networking is co-located.** Envoy's upstream `grpc-backend:host-gateway`
  resolves to the trainer container itself — exactly where the gRPC backend
  listens. Nothing about `:50051` needs to be published to the host.
- **Bind mounts just work.** The bundled compose mounts `envoy.yaml` and the
  certs dir from the installed `weightslab` package. The inner daemon shares the
  trainer container's filesystem, so those paths resolve with **no path
  alignment** required (contrast this with the siblings variant).
- We only re-publish the **browser-facing** ports (`5173`, `8080`) from the
  trainer container out to the host.

The trade-off: the container needs `--privileged`, and you pay for a nested
daemon (an isolated image store, a second pull of Envoy + the frontend image).

## Run it

```bash
# from this directory
docker compose up --build
```

Then open <http://localhost:5173>. Stop with `Ctrl+C` (or `docker compose down`).

### Use your local dev branch instead of PyPI

By default the image installs `weightslab` from PyPI. To build against the dev
branch (e.g. to pick up uncommitted-to-PyPI CLI changes):

```bash
docker compose build \
  --build-arg WEIGHTSLAB_SPEC="git+https://github.com/GrayboxTech/weightslab.git@dev"
docker compose up
```

(The repo must be reachable from the build — public, or pass build credentials.)

### 🔒 Secured TLS (HTTPS + mTLS)

DinD is the easiest option for TLS because the certs, Envoy, and the backend are
all co-located in one container. TLS is a **one-flag toggle** — just set
`WEIGHTSLAB_TLS=1`:

```bash
WEIGHTSLAB_TLS=1 docker compose up --build
```

then **trust the dev CA on your host** (step 2 below) and open
**https://localhost:5173**.

#### What the toggle does (in [entrypoint.sh](entrypoint.sh))

There are three pieces to a working TLS setup; the toggle handles the first two:

1. **Generate certs + configure the UI** — runs `weightslab ui launch --certs`,
   which generates the cert set into `WEIGHTSLAB_CERTS_DIR`
   (`/root/.weightslab-certs`) and configures Envoy + the frontend for HTTPS.
2. **Turn TLS on for the backend** — `--certs` only configures Envoy + the
   frontend, so the toggle also exports `GRPC_TLS_ENABLED=1` +
   `GRPC_TLS_CERT_DIR=$WEIGHTSLAB_CERTS_DIR` before the example. Without this,
   Envoy's upstream mTLS handshake fails against a plaintext backend (503s).

The generator writes `ca.crt`, `envoy-server.*` (browser↔Envoy), `envoy-client.*`
+ `ca.crt` (Envoy↔backend mTLS), and `backend-server.*` (the backend's own cert,
CN `host.docker.internal`). `GRPC_TLS_CERT_DIR` makes the backend pick up
`backend-server.*` + `ca.crt` from there.

> Optional: to also *enforce* the gRPC auth token the UI sends, export
> `GRPC_AUTH_TOKEN="$(cat $WEIGHTSLAB_CERTS_DIR/.grpc_auth_token)"` in the
> entrypoint (otherwise the token is simply ignored — TLS still encrypts).

#### 2. Trust the dev CA on the Windows host

The dev CA is generated *inside* the container, so your host browser doesn't
trust it yet. Export it and trust it (once):

```powershell
# pull the CA out of the running container
docker cp weightslab_trainer_dind:/root/.weightslab-certs/ca.crt .
# trust it for your user, then restart the browser
Import-Certificate -FilePath .\ca.crt -CertStoreLocation Cert:\CurrentUser\Root
```

The certs carry SANs for `localhost`/`127.0.0.1`, so `https://localhost:5173`
(frontend) and `https://localhost:8080` (Envoy) validate once the CA is trusted.
Then open **https://localhost:5173**. (Skip this and TLS still works, but the
browser shows a self-signed warning.)

> Without step 3 the connection is still encrypted, but the browser shows a
> self-signed warning. Without step 2, the UI loads over HTTPS but RPCs fail
> (Envoy can't complete mTLS to a plaintext backend).

## Notes / gotchas

- **`--privileged` is mandatory** for the inner `dockerd`. If your environment
  forbids it, use the siblings variant instead.
- First run is slow: it pulls `envoyproxy/envoy` and `graybx/weightslab`, then
  downloads MNIST for the example. The `dind_storage` volume caches images
  across runs.
- The example trains until stopped (`training_steps_to_do: null`) but starts
  paused (`is_training: false`); start/steer training from the UI.
