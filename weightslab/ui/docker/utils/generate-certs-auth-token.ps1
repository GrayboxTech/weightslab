param(
    [switch]$SkipTrust,
    [switch]$ForceCreateCerts
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command openssl -ErrorAction SilentlyContinue)) {
    throw "openssl is required but was not found in PATH."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Use user's default certificate directory
$userCertDir = Join-Path $env:USERPROFILE ".weightslab-certs"

New-Item -ItemType Directory -Force -Path $userCertDir | Out-Null

# Check if certs already exist in user directory
$certsExist = (Test-Path (Join-Path $userCertDir "ca.crt")) -and `
              (Test-Path (Join-Path $userCertDir "envoy-server.crt")) -and `
              (Test-Path (Join-Path $userCertDir "backend-server.crt"))

# If certs exist and not forcing recreation, skip generation
if ($certsExist -and -not $ForceCreateCerts) {
    Write-Host "Using existing certificates from $userCertDir..."
    exit 0
}

if ($ForceCreateCerts) {
    Write-Host "Force creating new certificates (-ForceCreateCerts)..."
}

$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("weightslab-certs-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

try {
    Write-Host "Generating local dev CA..."
    & openssl genrsa -out (Join-Path $tmpDir "ca.key") 4096
    & openssl req -x509 -new -nodes -key (Join-Path $tmpDir "ca.key") -sha256 -days 825 `
        -subj "/CN=weightslab-dev-ca" `
        -out (Join-Path $tmpDir "ca.crt")

    @"
    subjectAltName = DNS:localhost,IP:127.0.0.1,IP:0:0:0:0:0:0:0:1
extendedKeyUsage = serverAuth
"@ | Set-Content -Path (Join-Path $tmpDir "envoy-server.ext") -Encoding ASCII

    @"
    subjectAltName = DNS:localhost,DNS:host.docker.internal,IP:127.0.0.1,IP:0:0:0:0:0:0:0:1
extendedKeyUsage = serverAuth
"@ | Set-Content -Path (Join-Path $tmpDir "backend-server.ext") -Encoding ASCII

    @"
extendedKeyUsage = clientAuth
"@ | Set-Content -Path (Join-Path $tmpDir "envoy-client.ext") -Encoding ASCII

    Write-Host "Generating Envoy HTTPS server cert..."
    & openssl genrsa -out (Join-Path $tmpDir "envoy-server.key") 2048
    & openssl req -new -key (Join-Path $tmpDir "envoy-server.key") -subj "/CN=localhost" -out (Join-Path $tmpDir "envoy-server.csr")
    & openssl x509 -req -in (Join-Path $tmpDir "envoy-server.csr") -CA (Join-Path $tmpDir "ca.crt") -CAkey (Join-Path $tmpDir "ca.key") `
        -CAcreateserial -out (Join-Path $tmpDir "envoy-server.crt") -days 825 -sha256 -extfile (Join-Path $tmpDir "envoy-server.ext")

    Write-Host "Generating backend gRPC server cert..."
    & openssl genrsa -out (Join-Path $tmpDir "backend-server.key") 2048
    & openssl req -new -key (Join-Path $tmpDir "backend-server.key") -subj "/CN=host.docker.internal" -out (Join-Path $tmpDir "backend-server.csr")
    & openssl x509 -req -in (Join-Path $tmpDir "backend-server.csr") -CA (Join-Path $tmpDir "ca.crt") -CAkey (Join-Path $tmpDir "ca.key") `
        -CAcreateserial -out (Join-Path $tmpDir "backend-server.crt") -days 825 -sha256 -extfile (Join-Path $tmpDir "backend-server.ext")

    Write-Host "Generating Envoy mTLS client cert..."
    & openssl genrsa -out (Join-Path $tmpDir "envoy-client.key") 2048
    & openssl req -new -key (Join-Path $tmpDir "envoy-client.key") -subj "/CN=envoy-client" -out (Join-Path $tmpDir "envoy-client.csr")
    & openssl x509 -req -in (Join-Path $tmpDir "envoy-client.csr") -CA (Join-Path $tmpDir "ca.crt") -CAkey (Join-Path $tmpDir "ca.key") `
        -CAcreateserial -out (Join-Path $tmpDir "envoy-client.crt") -days 825 -sha256 -extfile (Join-Path $tmpDir "envoy-client.ext")

    Copy-Item -Force (Join-Path $tmpDir "ca.crt") (Join-Path $userCertDir "ca.crt")
    Copy-Item -Force (Join-Path $tmpDir "envoy-server.crt") (Join-Path $userCertDir "envoy-server.crt")
    Copy-Item -Force (Join-Path $tmpDir "envoy-server.key") (Join-Path $userCertDir "envoy-server.key")
    Copy-Item -Force (Join-Path $tmpDir "envoy-client.crt") (Join-Path $userCertDir "envoy-client.crt")
    Copy-Item -Force (Join-Path $tmpDir "envoy-client.key") (Join-Path $userCertDir "envoy-client.key")
    Copy-Item -Force (Join-Path $tmpDir "backend-server.crt") (Join-Path $userCertDir "backend-server.crt")
    Copy-Item -Force (Join-Path $tmpDir "backend-server.key") (Join-Path $userCertDir "backend-server.key")

    if (-not $SkipTrust) {
        Write-Host "Installing dev CA into CurrentUser Root trust store..."
        $caPath = Join-Path $userCertDir "ca.crt"
        $store = New-Object System.Security.Cryptography.X509Certificates.X509Store("Root", "CurrentUser")
        $store.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadWrite)
        try {
            $existing = $store.Certificates | Where-Object { $_.Subject -eq "CN=weightslab-dev-ca" }
            foreach ($cert in $existing) {
                $store.Remove($cert)
            }
            $certObj = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($caPath)
            $store.Add($certObj)
        }
        finally {
            $store.Close()
        }
        Write-Host "CA trusted for current Windows user."
    }
    else {
        Write-Host "Skipping CA trust installation (--SkipTrust)."
    }

    Write-Host "Dev certificates generated successfully."
    Write-Host "- Certs location: $userCertDir"
}
finally {
    if (Test-Path $tmpDir) {
        Remove-Item -Recurse -Force $tmpDir
    }
}
