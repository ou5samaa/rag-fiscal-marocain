# Serveur local au premier plan. Ctrl+C pour arrêter l'interface.
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$env:PYTHONHASHSEED='0'
$env:MLFLOW_DISABLE_TELEMETRY='true'
$env:MLFLOW_DISABLE_AGENT_HINT='1'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS='false'
Set-Location -LiteralPath $taskRoot
& (Join-Path $taskRoot '.venv/Scripts/python.exe') -B -m streamlit run (Join-Path $taskRoot 'streamlit_app.py') --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none --theme.primaryColor '#176b78' --theme.base light
exit $LASTEXITCODE
