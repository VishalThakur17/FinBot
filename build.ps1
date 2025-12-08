param(
    [switch]$Install,
    [switch]$Run
)

# Root of the repo (FinBot)
$Root = Split-Path $PSScriptRoot -Parent
$VenvPath = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$Requirements = Join-Path $PSScriptRoot "requirements.txt"

if ($Install) {
    Write-Host "Creating virtual environment at $VenvPath..."
    if (-not (Test-Path $VenvPython)) {
        python -m venv $VenvPath
    }

    Write-Host "Installing requirements..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r $Requirements
}

if ($Run) {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "Virtual environment not found. Run: .\build.ps1 -Install"
        exit 1
    }

    Write-Host "Starting FinRobot..."
    & $VenvPython server.py
}

