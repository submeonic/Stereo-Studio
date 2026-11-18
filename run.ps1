$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.\.venv\Scripts\python.exe')) { throw 'Run .\scripts\setup-windows.ps1 first.' }
& .\.venv\Scripts\python.exe -m wigglegram gui @args
exit $LASTEXITCODE
