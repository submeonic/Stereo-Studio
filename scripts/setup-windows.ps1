$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.12 with the py launcher, then retry.' }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
& .\.venv\Scripts\python.exe -m pip install -e '.[gui]'
if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host 'MP4 needs FFmpeg with libx264 on PATH. GIF and JPEG do not require it.'
}
Write-Host 'Ready. Start with .\run.ps1; select CPU SGBM.'
