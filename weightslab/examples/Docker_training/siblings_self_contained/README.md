# Training docker — Option C: self-contained siblings (no host bind mounts)

Like Option B (siblings / Docker-out-of-Docker) the trainer drives the **host**
docker daemon (mounted socket) so Envoy + frontend run as **siblings**. But this
variant **removes every host bind-mount dependency**, so it runs **entirely from
inside the container** — no `setup-host.sh`, no path alignment, and it works
**natively on Windows / Docker Desktop**.

Inside the trainer container, in order:

1. render Envoy's plaintext config from the installed `weightslab` package,
2. push it into a **named volume over the docker socket** (no host path),
3. bring up Envoy + frontend from a **bind-mount-free** [ui-compose.yml](ui-compose.yml),
4. `weightslab start example --cls` → serve the **gRPC backend** on `:50051`.

End result: open `http://localhost:5173` and the UI talks to training end-to-end.

## The trade-off vs Option B

Option B uses the stock **`weightslab ui launch`**, whose bundled compose
bind-mounts `envoy.yaml` + certs from the package — which the **host** daemon
must resolve, forcing path alignment + `setup-host.sh` (and breaking on Windows).

Option C instead **does not call `weightslab ui launch`**. It brings the UI up
itself from [ui-compose.yml](ui-compose.yml), which has **no host bind mounts**:

| What the bundled compose bind-mounts | How Option C delivers it without a host path |
|---|---|
| `envoy.yaml` (+ plaintext variants) | rendered from the installed package, then piped into the **`wl_envoy_cfg` named volume** over the socket (`docker run -i … busybox`) |
| `${WEIGHTSLAB_CERTS_DIR}` → Envoy/nginx | **omitted** — the frontend's nginx auto-falls back to **HTTP** when no certs are present; Envoy uses the fully-plaintext template |

It still reuses the **same images** (`envoyproxy/envoy`, `graybx/weightslab`) and
the **same Envoy template** that ships with weightslab, so it stays in sync.

> By default this path is **HTTP-only**. It *can* be secured with TLS using the
> same named-volume trick to deliver certs (no host bind mounts) — see
> [🔒 Enabling TLS](#-enabling-tls-https--mtls) below.

## How the wiring works

```
host browser
  → localhost:5173 ──► frontend container (sibling, HTTP, no certs)
  → localhost:8080 ──► Envoy container   (sibling, config from named volume)
                          │ grpc-backend:host-gateway → host:50051
                          ▼
                       host:50051 ──► trainer container's gRPC backend :50051
```

Same networking as Option B — the trainer **publishes `:50051`** so the sibling
Envoy reaches the in-container backend via `host-gateway`. Only the Envoy +
frontend publish `8080`/`5173` (from `ui-compose.yml`).

## ⚙️ Required configuration

| # | Requirement | Where | Why |
|---|---|---|---|
| 1 | **Mount `/var/run/docker.sock`** | [docker-compose.yml](docker-compose.yml) | drive the host daemon (siblings) |
| 2 | **Publish `50051:50051`** from the trainer | [docker-compose.yml](docker-compose.yml) | sibling Envoy dials it via `host-gateway` |
| 3 | **Stage Envoy config into a named volume over the socket** | [entrypoint.sh](entrypoint.sh) | replaces the host bind mount — no host path |
| 4 | **`ui-compose.yml` with no host bind mounts** (named volume + HTTP frontend) | [ui-compose.yml](ui-compose.yml) | nothing for the host daemon to resolve on disk |
| 5 | **`extra_hosts: grpc-backend:host-gateway`** on Envoy | [ui-compose.yml](ui-compose.yml) | route Envoy → host:50051 → trainer |
| 6 | **`GRPC_BACKEND_PORT=50051`** (matches the rendered Envoy config) | [docker-compose.yml](docker-compose.yml) / [Dockerfile](Dockerfile) | port the backend binds + Envoy dials |
| — | *Not needed (unlike Option B):* path alignment, `setup-host.sh`, a host source checkout | — | the daemon never resolves a host path |

### GPU — to run `nvidia-smi` / train on CUDA

| Requirement | Where |
|---|---|
| Host: NVIDIA driver **+ NVIDIA Container Toolkit** (`sudo nvidia-ctk runtime configure --runtime=docker`) | host setup |
| `deploy.resources.reservations.devices` (driver `nvidia`, `capabilities: [gpu]`) | [docker-compose.yml](docker-compose.yml) |
| `NVIDIA_VISIBLE_DEVICES=all` + `NVIDIA_DRIVER_CAPABILITIES=compute,utility` | [Dockerfile](Dockerfile) |

> The `deploy:` GPU block is a **hard requirement**: on a host with no NVIDIA
> GPU/toolkit, comment it out in [docker-compose.yml](docker-compose.yml) or
> `up` fails. Only the **trainer** needs the GPU.

## Run it

```bash
# from this directory — works on Windows/Docker Desktop, no host prep
docker compose up --build
```

Then open <http://localhost:5173>.

### Stopping

```bash
docker compose down                       # stops the trainer
# the sibling UI stack is a separate project:
docker compose -p weightslab_ui -f ui-compose.yml down
docker volume rm wl_envoy_cfg             # optional: drop the staged config
```

## 🔒 Enabling TLS (HTTPS + mTLS)

TLS works here too, **still with no host bind mounts** — the certs are delivered
the same way as `envoy.yaml`: piped into named volumes over the docker socket.
It's a **one-flag toggle**:

```bash
WEIGHTSLAB_TLS=1 docker compose up --build
```

then **trust the dev CA on your host** (step 2 below) and open
**https://localhost:5173**.

#### What the toggle does (in [entrypoint.sh](entrypoint.sh))

In TLS mode the entrypoint generates one cert set with `weightslab se` (writes
`ca.crt`, `envoy-server.*`, `envoy-client.*`, `backend-server.*` +
`.grpc_auth_token` into `WEIGHTSLAB_CERTS_DIR`) and wires up four layers — all
fed over the socket, no host paths:

| Layer | Needs | Delivered via |
|---|---|---|
| Browser ↔ Envoy (downstream) | `envoy-server.crt/key` | `wl_envoy_cfg` volume → `/etc/envoy/certs` |
| Envoy ↔ backend (upstream mTLS) | `envoy-client.crt/key` + `ca.crt` | `wl_envoy_cfg` volume → `/etc/envoy/certs` |
| Browser ↔ frontend (nginx HTTPS) | `envoy-server.crt/key` | `wl_nginx_certs` volume → `/etc/nginx/certs` |
| Backend gRPC server | `backend-server.crt/key` + `ca.crt` | `GRPC_TLS_ENABLED=1` + `GRPC_TLS_CERT_DIR` |

Concretely, vs the HTTP path, the toggle:

1. runs `weightslab se` and renders the **full mTLS** template
   (`ui/envoy/envoy.yaml`, not the `*_plaintext` one) into `wl_envoy_cfg`;
2. stages the Envoy certs into `wl_envoy_cfg:/etc/envoy/certs` and the frontend
   cert into `wl_nginx_certs` (and `chmod a+rX` them — the generated keys are
   `0600`/root, and Envoy/nginx run non-root, so without this Envoy crashes with
   *"unable to read file …envoy-client.key"*);
3. sets `WS_SERVER_PROTOCOL=https` so the frontend's nginx serves HTTPS
   ([ui-compose.yml](ui-compose.yml) always mounts `wl_nginx_certs`, empty in
   HTTP mode);
4. exports `GRPC_TLS_ENABLED=1` + `GRPC_TLS_CERT_DIR=$WEIGHTSLAB_CERTS_DIR` so
   the backend serves gRPC over TLS (else Envoy's upstream mTLS fails → 503s).

> Optional: to *enforce* the gRPC auth token, also export
> `GRPC_AUTH_TOKEN="$(cat $WEIGHTSLAB_CERTS_DIR/.grpc_auth_token)"` (otherwise the
> token the UI sends is ignored — TLS still encrypts).

#### 2. Trust the dev CA on the Windows host

The dev CA is generated *inside* the container, so trust it on the host once:

```powershell
docker cp weightslab_trainer_selfcontained:/root/.weightslab-certs/ca.crt .
Import-Certificate -FilePath .\ca.crt -CertStoreLocation Cert:\CurrentUser\Root
```

Then open **https://localhost:5173**. Certs carry `localhost`/`127.0.0.1` SANs, so
the frontend (5173) and Envoy (8080) both validate once the CA is trusted.
(Skip this and TLS still works, but the browser shows a self-signed warning.)

## Notes

- First run pulls `envoyproxy/envoy` + `graybx/weightslab` into the host image
  cache and downloads MNIST for the example.
- HTTP by default; TLS is opt-in (section above).
- The example starts paused (`is_training: false`); start/steer it from the UI.
