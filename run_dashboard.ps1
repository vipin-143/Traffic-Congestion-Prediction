$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localInterpreter = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$workspaceInterpreter = Join-Path $PSScriptRoot '../../work/project-env/Scripts/python.exe'
if (Test-Path -LiteralPath $localInterpreter) { & $localInterpreter -m streamlit run app.py }
elseif (Test-Path -LiteralPath $workspaceInterpreter) { & $workspaceInterpreter -m streamlit run app.py }
else { Write-Host 'Create .venv and install requirements.txt first. See README.md.' }
