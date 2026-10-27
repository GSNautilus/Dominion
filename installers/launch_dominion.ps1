# Launch DOMINION. Double-click "Launch DOMINION.bat" (or the Desktop icon).
# Opens napari with a file-open dialog so you can pick an image.

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

$conda = Get-CondaExe
if (-not $conda) {
    Write-Host "DOMINION isn't installed yet."
    Write-Host "Run 'Install DOMINION (Basic).bat' or 'Install DOMINION (Full).bat' first."
    Read-Host "Press Enter to close"
    exit 1
}

# --no-capture-output lets napari's console messages stream through.
& $conda run --no-capture-output -n dominion dominion
