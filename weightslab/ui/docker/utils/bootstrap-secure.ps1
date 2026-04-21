param(
    [switch]$no_auth_token,
    [switch]$force_create_certs,
    [switch]$no_cert,
    [switch]$dev
)

$ErrorActionPreference = "Stop"

function New-StrongHexToken {
    param(
        [int]$ByteCount = 32
    )

    $bytes = [byte[]]::new($ByteCount)
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)

    $hashBytes = [System.Security.Cryptography.SHA256]::HashData($bytes)
    return [Convert]::ToHexString($hashBytes).ToLowerInvariant()
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker is required but was not found in PATH."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location $scriptDir
try {
    if (-not $no_cert) {
        Write-Host "[1/3] Generating development certificates..."
        $certArgs = @()
        if ($force_create_certs) {
            $certArgs += "-force_create_certs"
        }
        & .\generate-certs.ps1 @certArgs
    } else {
        Write-Host "[1/3] Skipping certificate generation (-no_cert flag set)..."
    }

    Write-Host "[2/3] Exporting secure backend defaults for current shell..."
    $env:GRPC_TLS_ENABLED = "1"
    $env:GRPC_TLS_REQUIRE_CLIENT_AUTH = "1"
    $env:GRPC_TLS_CERT_FILE = (Join-Path $env:USERPROFILE ".weightslab-certs/backend-server.crt")
    $env:GRPC_TLS_KEY_FILE = (Join-Path $env:USERPROFILE ".weightslab-certs/backend-server.key")
    $env:GRPC_TLS_CA_FILE = (Join-Path $env:USERPROFILE ".weightslab-certs/ca.crt")
    $env:ENVOY_DOWNSTREAM_TLS = "on"
    $env:ENVOY_UPSTREAM_TLS = "on"
    $env:WEIGHTSLAB_CERTS_DIR = (Join-Path $env:USERPROFILE ".weightslab-certs")
    $env:WS_SERVER_PROTOCOL = "https"
    $env:VITE_SERVER_PROTOCOL = "https"
    $env:VITE_DEV_SERVER_HTTPS = "1"
    $env:VITE_DEV_SERVER_CERT_FILE = "/app/envoy/certs/envoy-server.crt"
    $env:VITE_DEV_SERVER_KEY_FILE = "/app/envoy/certs/envoy-server.key"

    # Set WL_ENABLE_GRPC_AUTH_TOKEN based on -no_auth_token flag (default enabled)
    if ($no_auth_token) {
        $env:WL_ENABLE_GRPC_AUTH_TOKEN = "0"
        $env:VITE_WL_ENABLE_GRPC_AUTH_TOKEN = "0"
    } else {
        $env:WL_ENABLE_GRPC_AUTH_TOKEN = "1"
        $env:VITE_WL_ENABLE_GRPC_AUTH_TOKEN = "1"
    }

    $enableGrpcAuthToken = [string]::Equals(([string]$env:WL_ENABLE_GRPC_AUTH_TOKEN).Trim(), "1", [System.StringComparison]::OrdinalIgnoreCase)
    if ($enableGrpcAuthToken) {
        if ([string]::IsNullOrWhiteSpace($env:GRPC_AUTH_TOKEN)) {
            $env:GRPC_AUTH_TOKEN = New-StrongHexToken
            Write-Host "Generated GRPC_AUTH_TOKEN for this shell session."
        }
        else {
            Write-Host "Using existing GRPC_AUTH_TOKEN from current shell session."
        }
        $env:VITE_GRPC_AUTH_TOKEN = $env:GRPC_AUTH_TOKEN
    }
    else {
        Remove-Item Env:GRPC_AUTH_TOKEN -ErrorAction SilentlyContinue
        Remove-Item Env:VITE_GRPC_AUTH_TOKEN -ErrorAction SilentlyContinue
        Write-Host "GRPC auth token disabled (-no_auth_token flag set)."
    }

    if (-not $no_cert) {
        Push-Location ".."
        try {
            # Export variables for docker-compose/docker build
            $env:DOCKER_BUILDKIT = "1"

            if ($dev) {
                Write-Host "[3/3] Starting secured Dev Envoy + frontend stack..."
                & docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --force-recreate envoy
                & docker compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache weights_studio
                & docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --force-recreate weights_studio
                Write-Host "Secure dev stack is up."
            } else {
                Write-Host "[3/3] Starting secured Production ready Envoy + frontend stack..."
                & docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate envoy
                & docker compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache weights_studio
                & docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate weights_studio
                Write-Host "Secure prod stack is up."
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "[3/3] Skipping stack startup (-no_cert flag set, only updating auth token)..."
    }

    Write-Host "Backend TLS env exported in this shell. Start backend from this shell to use them."
    if ($enableGrpcAuthToken) {
        Write-Host "GRPC_AUTH_TOKEN is set in this shell for backend gRPC auth."
    }

    Write-Host ""
    Write-Host "Available flags:"
    Write-Host "  -no_auth_token          Disable gRPC auth token"
    Write-Host "  -force_create_certs     Recreate certificates even if they exist"
    Write-Host "  -no_cert                Skip certificate generation/copying, only update auth token"
    Write-Host "  -dev                    Use dev docker-compose (default: prod)"
}
finally {
    Pop-Location
}
