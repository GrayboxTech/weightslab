# Default environment file path
$envFilePath = ".\.env"

# Check if the file exists
if (-not (Test-Path $envFilePath)) {
    Write-Error "Environment file '$envFilePath' not found."
    exit 1
}

Write-Host "Loading variables from $envFilePath..."

# Read the content of the file
$lines = Get-Content -Path $envFilePath

foreach ($line in $lines) {
    # Cleanup: ignore empty lines or comments (#)
    if ($line.Trim() -match '^\s*(#.*)?$') {
        continue
    }

    # Use a regular expression to separate the key and the value
    if ($line -match '^([^=]+)=(.*)$') {
        $key = $Matches[1].Trim()
        $value = $Matches[2].Trim()

        # Cleanup: remove quotes around the value (if present)
        $value = $value.Trim('"', "'")

        # Set the variable in the current PowerShell session environment
        Set-Item -Path Env:\$key -Value $value

        Write-Host "   - Set: $($key)"
    }
}

Write-Host "Loading completed."