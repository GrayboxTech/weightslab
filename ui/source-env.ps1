# Nom du fichier d'environnement par défaut
$envFilePath = ".\.env"

# Vérifie si le fichier existe
if (-not (Test-Path $envFilePath)) {
    Write-Error "Fichier d'environnement '$envFilePath' introuvable."
    exit 1
}

Write-Host "Chargement des variables à partir de $envFilePath..."

# Lit le contenu du fichier
$lines = Get-Content -Path $envFilePath

foreach ($line in $lines) {
    # Nettoyage : Ignorer les lignes vides ou les commentaires (#)
    if ($line.Trim() -match '^\s*(#.*)?$') {
        continue
    }

    # Utilise une expression régulière pour séparer la clé et la valeur
    if ($line -match '^([^=]+)=(.*)$') {
        $key = $Matches[1].Trim()
        $value = $Matches[2].Trim()

        # Nettoyage : Retire les guillemets de la valeur (si présents)
        $value = $value.Trim('"', "'")

        # Définit la variable dans l'environnement de la session PowerShell actuelle
        Set-Item -Path Env:\$key -Value $value

        Write-Host "   - Définie: $($key)"
    }
}

Write-Host "Chargement terminé."