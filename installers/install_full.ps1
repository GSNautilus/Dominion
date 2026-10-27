# DOMINION - Full installer (adds StarDist nuclei mode).
# Double-click "Install DOMINION (Full).bat" to run this.

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

$repo = Split-Path -Parent $PSScriptRoot

Write-Host "======================================"
Write-Host "  DOMINION (Full) installer"
Write-Host "  Includes StarDist nuclei detection."
Write-Host "  Uses TensorFlow 2.10; an NVIDIA GPU"
Write-Host "  is strongly recommended (CPU is slow)."
Write-Host "======================================"
Write-Host ""

$conda = Get-OrInstallConda

# Newer conda refuses the Anaconda "defaults" channel non-interactively until
# its Terms of Service are accepted. Best-effort; ignored on older conda that
# lacks the subcommand.
Write-Host "Accepting Anaconda channel Terms of Service (if required)..."
& $conda tos accept --override-channels `
    --channel "https://repo.anaconda.com/pkgs/main" `
    --channel "https://repo.anaconda.com/pkgs/r" | Out-Null

New-DominionEnv -Conda $conda -EnvFile (Join-Path $repo 'environment-full.yml')

Write-Host "Installing DOMINION + nuclei extras into the environment..."
# TensorFlow is already provided by conda above; the [nuclei] extra adds
# stardist and csbdeep and leaves the existing tensorflow in place.
& $conda run -n dominion python -m pip install "$($repo)[nuclei]"
if ($LASTEXITCODE -ne 0) { throw "pip install of DOMINION[nuclei] failed." }

Set-PyQt5 -Conda $conda
New-DesktopShortcut -Name 'DOMINION' -Target (Join-Path $PSScriptRoot 'Launch DOMINION.bat')

Write-Host ""
Write-Host "Done! Start DOMINION from the 'DOMINION' icon on your Desktop."
Write-Host "Both signal mode and nuclei mode are available."
