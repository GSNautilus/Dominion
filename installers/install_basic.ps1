# DOMINION - Basic installer (signal mode; no TensorFlow / GPU).
# Double-click "Install DOMINION (Basic).bat" to run this.

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_common.ps1')

$repo = Split-Path -Parent $PSScriptRoot

Write-Host "======================================"
Write-Host "  DOMINION (Basic) installer"
Write-Host "  Default signal-mode pipeline."
Write-Host "  No GPU or TensorFlow required."
Write-Host "======================================"
Write-Host ""

$conda = Get-OrInstallConda
New-DominionEnv -Conda $conda -EnvFile (Join-Path $repo 'environment-basic.yml')

Write-Host "Installing DOMINION into the environment..."
& $conda run -n dominion python -m pip install "$repo"
if ($LASTEXITCODE -ne 0) { throw "pip install of DOMINION failed." }

Set-PyQt5 -Conda $conda
New-DesktopShortcut -Name 'DOMINION' -Target (Join-Path $PSScriptRoot 'Launch DOMINION.bat')

Write-Host ""
Write-Host "Done! Start DOMINION from the 'DOMINION' icon on your Desktop,"
Write-Host "or from 'installers\Launch DOMINION.bat'."
Write-Host "A file-open dialog will ask you to pick an image."
