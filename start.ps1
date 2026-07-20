[CmdletBinding()]
param(
    [string]$HostAddress = '127.0.0.1',
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root '.runtime\environment\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'The local runtime is missing. Run install.ps1 first.' }
if (-not (Test-Path -LiteralPath (Join-Path $Root '.runtime-assets\model-install-receipt.json'))) { throw 'Model assets are missing. Run install.ps1 first.' }

$env:PYTHONNOUSERSITE = '1'
$env:PYTHONSAFEPATH = '1'
$env:HF_HOME = Join-Path $Root '.runtime-assets\huggingface'
$env:OSU_AI_WEB_PROFILE = 'audio-review'
$env:OSU_AI_WHITEBOX_VENDOR_ROOT = Join-Path $Root 'vendor\Mapperatorinator'

if (-not $NoBrowser) {
    Start-Job -ScriptBlock { param($Url) Start-Sleep -Seconds 3; Start-Process $Url } -ArgumentList "http://$HostAddress`:$Port" | Out-Null
}
& $Python -I -m osu_ai_detector.web_cli --host $HostAddress --port $Port --workers 1 --log-level info
