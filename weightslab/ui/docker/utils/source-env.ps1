param(
    [string]$EnvFilePath = ".\..\.env"
)

# Default environment file path (or path provided as argument)
$envFilePath = $EnvFilePath

# Check that the file exists
if (-not (Test-Path $envFilePath)) {
    Write-Error "Environment file '$envFilePath' not found."
    exit 1
}

Write-Host "Loading environment variables from $envFilePath..."

# Read file content
$lines = Get-Content -Path $envFilePath

foreach ($line in $lines) {
    # Ignore empty lines and comments (#)
    if ($line.Trim() -match '^\s*(#.*)?$') {
        continue
    }

    # Use regex to split key/value pairs
    if ($line -match '^([^=]+)=(.*)$') {
        $key = $Matches[1].Trim()
        $value = $Matches[2].Trim()

        # Remove surrounding quotes if present
        $value = $value.Trim('"', "'")

        # Set environment variable in current PowerShell session
        Set-Item -Path Env:\$key -Value $value

        Write-Host "   - Set: $($key)"
    }
}

Write-Host "Environment loading completed."