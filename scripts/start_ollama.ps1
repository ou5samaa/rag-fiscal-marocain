param([switch]$Stop,[switch]$CPU)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$executable=Join-Path $taskRoot 'runtime/ollama/0.40.1/ollama.exe'
$sessionDir=Join-Path $taskRoot 'runtime/ollama/session'
$stateFile=Join-Path $sessionDir 'server.json'
$endpoint='http://127.0.0.1:11435'
if(Test-Path -LiteralPath $stateFile){
    $state=Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    $running=Get-Process -Id $state.pid -ErrorAction SilentlyContinue
    if($running){
        if($running.Path -ne $executable -or $running.StartTime.ToUniversalTime().Ticks -ne ([datetime]$state.started_at).ToUniversalTime().Ticks){throw 'PID réutilisé : ne pas arrêter un autre processus.'}
        if($Stop){
            # Le runner GPU est un enfant distinct ; Stop-Process ne termine pas l'arbre sous Windows.
            $runnerPath=Join-Path $taskRoot 'runtime/ollama/0.40.1/lib/ollama/llama-server.exe'
            $ownedChildren=@(Get-CimInstance Win32_Process | Where-Object {$_.ParentProcessId -eq $running.Id -and $_.ExecutablePath -eq $runnerPath} | ForEach-Object {
                $child=Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
                if($child){[pscustomobject]@{pid=$child.Id;path=$child.Path;ticks=$child.StartTime.ToUniversalTime().Ticks}}
            })
            Stop-Process -Id $running.Id
            foreach($owned in $ownedChildren){
                $live=Get-Process -Id $owned.pid -ErrorAction SilentlyContinue
                if($live -and $live.Path -eq $owned.path -and $live.StartTime.ToUniversalTime().Ticks -eq $owned.ticks){Stop-Process -Id $live.Id}
            }
            Remove-Item -LiteralPath $stateFile
            Write-Output 'Serveur du projet et ses processus GPU arrêtés.'; exit
        }
        if($CPU -and -not $state.cpu_mode){throw 'Serveur déjà actif sans mode CPU enregistré. Arrêter avec -Stop puis relancer avec -CPU.'}
        Write-Output "Serveur déjà actif : $endpoint"; exit
    }
}
if($Stop){Write-Output 'Serveur du projet déjà arrêté.'; exit}
if(-not (Test-Path -LiteralPath $executable)){throw 'Exécuter scripts/install_ollama.py au préalable.'}
try {$null=Invoke-RestMethod "$endpoint/api/version" -TimeoutSec 2; throw 'Port 11435 déjà occupé par un serveur non identifié.'}
catch {if($_.Exception.Message -like 'Port 11435*'){throw}}
New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null
$env:OLLAMA_HOST='127.0.0.1:11435'
$env:OLLAMA_MODELS=Join-Path $taskRoot 'models/ollama'
$env:OLLAMA_NO_CLOUD='1'
$env:OLLAMA_NUM_PARALLEL='1'
$env:OLLAMA_MAX_LOADED_MODELS='1'
$env:OLLAMA_CONTEXT_LENGTH='8192'
$env:OLLAMA_KEEP_ALIVE='5m'
if($CPU){
    $env:CUDA_VISIBLE_DEVICES='-1'
    $env:GGML_VK_VISIBLE_DEVICES='-1'
    $env:OLLAMA_VULKAN='0'
}
$proc=Start-Process -FilePath $executable -ArgumentList 'serve' -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $sessionDir 'stdout.log') -RedirectStandardError (Join-Path $sessionDir 'stderr.log')
@{pid=$proc.Id;started_at=$proc.StartTime.ToUniversalTime().ToString('o');endpoint=$endpoint;executable=$executable;cpu_mode=[bool]$CPU} | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding utf8
for($attempt=0;$attempt -lt 30;$attempt++){
    Start-Sleep -Milliseconds 500
    try {$v=Invoke-RestMethod "$endpoint/api/version" -TimeoutSec 1; Write-Output "Ollama $($v.version) disponible : $endpoint"; exit}
    catch {if($proc.HasExited){throw 'Ollama a quitté : consulter runtime/ollama/session/stderr.log.'}}
}
throw 'Le serveur ne répond pas encore ; consulter les journaux.'
