# Interface de suivi locale, au premier plan : Ctrl+C pour l'arrêter.
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$python=Join-Path $taskRoot '.venv/Scripts/python.exe'
$trackingDir=Join-Path $taskRoot 'data/mlflow'
New-Item -ItemType Directory -Path $trackingDir -Force | Out-Null
$database=(Join-Path $taskRoot 'data/mlflow/tracking.db').Replace('\','/')
$env:MLFLOW_DISABLE_TELEMETRY='true'
$env:MLFLOW_DISABLE_AGENT_HINT='true'
& $python -B -m mlflow ui --backend-store-uri "sqlite:///$database" --host 127.0.0.1 --port 5000 --workers 1
exit $LASTEXITCODE
