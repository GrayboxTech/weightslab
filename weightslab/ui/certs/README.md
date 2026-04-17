# TLS certificate layout for packaged UI Envoy

Place the same files as `weights_studio/envoy/certs` in this directory when using the packaged docker-compose:

For local development, generate them from `weights_studio/docker` with:

- PowerShell: `./generate-dev-certs.ps1`
- Bash: `./generate-dev-certs.sh`

- `envoy-server.crt`
- `envoy-server.key`
- `envoy-client.crt`
- `envoy-client.key`
- `ca.crt`

Do not commit private keys.
