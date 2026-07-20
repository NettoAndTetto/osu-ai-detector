[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$SkipBrowserLaunch
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Spec = Get-Content -LiteralPath (Join-Path $Root 'release-spec.json') -Raw | ConvertFrom-Json
$RuntimeRoot = Join-Path $Root '.runtime'
$MiniforgeRoot = Join-Path $RuntimeRoot 'miniforge'
$EnvironmentRoot = Join-Path $RuntimeRoot 'environment'
$DownloadRoot = Join-Path $RuntimeRoot 'downloads'
$Installer = Join-Path $DownloadRoot $Spec.miniforge.filename
$ChecksumFile = "$Installer.sha256"

New-Item -ItemType Directory -Force -Path $DownloadRoot | Out-Null
if (-not (Test-Path -LiteralPath $Installer)) {
    Write-Host 'Downloading pinned Miniforge installer...'
    Invoke-WebRequest -UseBasicParsing -Uri $Spec.miniforge.url -OutFile $Installer
    Invoke-WebRequest -UseBasicParsing -Uri $Spec.miniforge.sha256_url -OutFile $ChecksumFile
}
$Expected = ((Get-Content -LiteralPath $ChecksumFile -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
$Actual = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Expected -ne $Actual) { throw 'Miniforge installer SHA-256 mismatch.' }

if (-not (Test-Path -LiteralPath (Join-Path $MiniforgeRoot 'Scripts\conda.exe'))) {
    Write-Host 'Installing isolated Miniforge runtime...'
    $Process = Start-Process -FilePath $Installer -ArgumentList @('/InstallationType=JustMe','/RegisterPython=0','/S',"/D=$MiniforgeRoot") -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "Miniforge installer failed with exit code $($Process.ExitCode)." }
}

$Conda = Join-Path $MiniforgeRoot 'Scripts\conda.exe'
if (-not (Test-Path -LiteralPath (Join-Path $EnvironmentRoot 'python.exe'))) {
    Write-Host 'Creating the pinned application environment...'
    & $Conda env create --prefix $EnvironmentRoot --file (Join-Path $Root 'environment-release.yml') --yes
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
}

$Python = Join-Path $EnvironmentRoot 'python.exe'
$env:HF_HOME = Join-Path $Root '.runtime-assets\huggingface'
Write-Host 'Downloading and verifying calibrated and upstream model assets...'
& $Python (Join-Path $Root 'scripts\download_release_models.py') --app-root $Root --hf-home $env:HF_HOME
if ($LASTEXITCODE -ne 0) { throw 'Model asset installation failed.' }

$Git = Join-Path $EnvironmentRoot 'Library\bin\git.exe'
$Vendor = Join-Path $Root 'vendor\Mapperatorinator'
if (-not (Test-Path -LiteralPath (Join-Path $Vendor '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Vendor) | Out-Null
    & $Git clone $Spec.upstream_source.repo $Vendor
    if ($LASTEXITCODE -ne 0) { throw 'Mapperatorinator source download failed.' }
}
& $Git -C $Vendor checkout --detach $Spec.upstream_source.revision
if ($LASTEXITCODE -ne 0) { throw 'Mapperatorinator source revision verification failed.' }

$Receipt = @{
    schema_version = 1
    release = $Spec.tag
    installed_at_utc = [DateTime]::UtcNow.ToString('o')
    miniforge_sha256 = $Actual
    cpu_only_requested = [bool]$CpuOnly
} | ConvertTo-Json
$Receipt | Set-Content -LiteralPath (Join-Path $RuntimeRoot 'installation-receipt.json') -Encoding UTF8
Write-Host 'Installation complete. Run .\start.ps1 to launch the local application.'
