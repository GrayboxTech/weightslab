# Running WeightsLab from a "training docker" — three options

Each option runs the WeightsLab UI stack + the classification training inside a
container so the example ends up reachable at <http://localhost:5173>, talking to
the training process end-to-end. They differ in **how the container gets a docker
daemon** to launch Envoy + frontend, and the configuration that follows.

| | [docker_in_docker](docker_in_docker/) (A · DinD) | [siblings_docker](siblings_docker/) (B · DooD) | [siblings_self_contained](siblings_self_contained/) (C) |
|---|---|---|---|
| Docker daemon | own daemon, nested | host daemon (socket) | host daemon (socket) |
| Envoy/frontend run | nested in the trainer | siblings on the host | siblings on the host |
| Starts the UI via | `weightslab ui launch` | `weightslab ui launch` | own bind-mount-free `ui-compose.yml` |
| `--privileged` | **required** | no | no |
| gRPC `:50051` | not published | **published to host** | **published to host** |
| Browser ports 5173/8080 | re-published by trainer | published by siblings | published by siblings |
| Host bind mounts | resolve in-container — none | **path alignment** required | **none** (config via named volume + HTTP) |
| Host setup | none | `setup-host.sh` (clone to aligned path) | none |
| TLS (HTTPS) | ✅ `--certs` (+ backend TLS env) | ✅ `--certs` (aligned certs) | ✅ opt-in (certs via named volume) |
| Windows / Docker Desktop | ✅ works | ⚠️ awkward (use WSL2) | ✅ works natively |
| Best when | want isolation / Windows / TLS | Linux host, stock `ui launch` | "all from inside", no host prep, Windows |

See each directory's `README.md` for the detailed wiring diagram and run steps.

- **A (DinD)** — fully self-contained via a nested daemon; needs `--privileged`.
- **B (siblings)** — uses the stock `weightslab ui launch`; its bundled compose
  bind-mounts the Envoy config + certs, so the host must see those files (path
  alignment + `setup-host.sh`). Cleanest on a Linux host.
- **C (self-contained siblings)** — siblings *without* host bind mounts: it
  delivers the Envoy config (and, for TLS, the certs) through named volumes over
  the socket, so no host prep and it works natively on Windows. The trade-off:
  it doesn't use `weightslab ui launch`. HTTP by default, TLS opt-in.

> All three can run secured TLS — see each README's "🔒" section. **A and C
> expose a one-flag toggle:** `WEIGHTSLAB_TLS=1 docker compose up --build`. Note
> that for **every** option cert generation only configures Envoy + the frontend;
> the **backend** example also needs `GRPC_TLS_ENABLED=1` + `GRPC_TLS_CERT_DIR`
> (the toggle handles this), or Envoy's upstream mTLS fails. And the dev CA must
> be trusted on the host browser to avoid a self-signed warning.
