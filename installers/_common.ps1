# Shared helpers for the DOMINION installers (dot-sourced by the install /
# launch / update scripts). Written for Windows PowerShell 5.1 - no &&, no
# ternary, no null-coalescing.

function Get-CondaExe {
    # Return the path to a usable conda.exe, or $null if none is found.
    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $roots = @(
        (Join-Path $env:USERPROFILE 'miniforge3'),
        (Join-Path $env:USERPROFILE 'miniconda3'),
        (Join-Path $env:USERPROFILE 'anaconda3'),
        (Join-Path $env:LOCALAPPDATA 'miniforge3'),
        (Join-Path $env:ProgramData  'miniforge3')
    )
    foreach ($r in $roots) {
        $p = Join-Path $r 'Scripts\conda.exe'
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Install-Miniforge {
    # Download and silently install Miniforge (a minimal conda), then return
    # the path to its conda.exe. One-time, ~100 MB.
    $dest = Join-Path $env:USERPROFILE 'miniforge3'
    $url = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe'
    $installer = Join-Path $env:TEMP 'Miniforge3-Windows-x86_64.exe'

    Write-Host "No conda found. Downloading Miniforge (one-time)..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $installer

    Write-Host "Installing Miniforge to $dest (this takes a minute)..."
    # NSIS-style silent install; /D= must be last and unquoted, so pass the
    # whole argument string verbatim rather than as an array.
    Start-Process -FilePath $installer `
        -ArgumentList "/InstallationType=JustMe /RegisterPython=0 /S /D=$dest" -Wait

    $conda = Join-Path $dest 'Scripts\conda.exe'
    if (-not (Test-Path $conda)) {
        throw "Miniforge installation failed: conda.exe not found at $conda"
    }
    return $conda
}

function Get-OrInstallConda {
    $conda = Get-CondaExe
    if ($conda) {
        Write-Host "Using existing conda: $conda"
        return $conda
    }
    return Install-Miniforge
}

function New-DominionEnv {
    # (Re)create the 'dominion' conda environment from an environment file.
    param(
        [Parameter(Mandatory = $true)][string]$Conda,
        [Parameter(Mandatory = $true)][string]$EnvFile
    )
    Write-Host "Preparing the 'dominion' environment (this can take a few minutes)..."
    # Remove any previous environment of the same name so re-running is clean.
    # This errors harmlessly if it doesn't exist yet; we ignore that.
    & $Conda env remove -n dominion -y | Out-Null
    & $Conda env create -f $EnvFile
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the conda environment from $EnvFile"
    }
}

function Set-PyQt5 {
    # napari may install PyQt6, whose DLLs fail to load on some Windows
    # machines ("QtBindingsNotFoundError"). PyQt5 is the reliable default.
    param([Parameter(Mandatory = $true)][string]$Conda)
    Write-Host "Setting a Windows-reliable Qt backend (PyQt5)..."
    & $Conda run -n dominion python -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip | Out-Null
    & $Conda run -n dominion python -m pip install PyQt5 | Out-Null
}

function New-DesktopShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Target
    )
    $desktop = [Environment]::GetFolderPath('Desktop')
    $lnk = Join-Path $desktop ($Name + '.lnk')
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($lnk)
    $sc.TargetPath = $Target
    $sc.WorkingDirectory = (Split-Path -Parent $Target)
    $sc.IconLocation = 'shell32.dll,25'
    $sc.Save()
    Write-Host "Created Desktop shortcut: $Name"
}
