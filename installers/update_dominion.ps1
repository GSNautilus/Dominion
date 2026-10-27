# Update DOMINION to the latest version. Double-click "Update DOMINION.bat".
# Pulls the latest code (if this is a git checkout) and reinstalls it into the
# existing 'dominion' environment, keeping whichever flavour (Basic/Full) you
# have.

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

$repo = Split-Path -Parent $PSScriptRoot

$conda = Get-CondaExe
if (-not $conda) {
    Write-Host "DOMINION isn't installed yet - run an installer first."
    Read-Host "Press Enter to close"
    exit 1
}

# 1. Get the latest code.
if (Test-Path (Join-Path $repo '.git')) {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        Write-Host "Pulling the latest code..."
        & git -C $repo pull
    } else {
        Write-Host "Git isn't installed, so the code can't auto-update."
        Write-Host "Download the latest ZIP from GitHub and re-run the installer."
    }
} else {
    Write-Host "This isn't a git checkout, so the code can't auto-update."
    Write-Host "Download the latest ZIP from GitHub and re-run the installer."
}

# 2. Reinstall into the env, preserving Full vs. Basic. Detect whether the
#    nuclei stack (TensorFlow) is present and reinstall with the matching
#    extra so dependencies stay consistent.
& $conda run -n dominion python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('tensorflow') else 1)"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Updating DOMINION (Full)..."
    & $conda run -n dominion python -m pip install --upgrade "$($repo)[nuclei]"
} else {
    Write-Host "Updating DOMINION (Basic)..."
    & $conda run -n dominion python -m pip install --upgrade "$repo"
}
if ($LASTEXITCODE -ne 0) { throw "Update failed during pip install." }

Write-Host ""
Write-Host "Update complete."
Read-Host "Press Enter to close"
