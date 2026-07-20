[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)][string]$OutputDirectory = '.\offline-output'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputDirectory = [System.IO.Path]::GetFullPath((Join-Path $Root $OutputDirectory))
$Work = Join-Path $OutputDirectory 'osu-ai-detector-v1.0.0-offline'
if (Test-Path -LiteralPath $Work) { throw "Output already exists: $Work" }
New-Item -ItemType Directory -Force -Path $Work | Out-Null

Write-Host 'Copying the allowlisted application release...'
Get-ChildItem -LiteralPath $Root -Force | Where-Object { $_.Name -notin @('.git','.runtime','.runtime-assets','manual-models','offline-output') } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $Work -Recurse -Force
}

Write-Host 'Creating a connected-machine runtime and downloading pinned assets...'
& (Join-Path $Work 'install.ps1') -SkipBrowserLaunch
if ($LASTEXITCODE -ne 0) { throw 'Connected-machine asset preparation failed.' }

$Conda = Join-Path $Work '.runtime\miniforge\Scripts\conda.exe'
$Environment = Join-Path $Work '.runtime\environment'
$EnvironmentPython = Join-Path $Environment 'python.exe'
# Replace the online editable install with a self-contained wheel install before
# packing.  By this point the verified model JSON files are present under the
# source package and are included by pyproject.toml package-data.
$BuildDirectory = Join-Path $Work 'build'
if (Test-Path -LiteralPath $BuildDirectory) { Remove-Item -LiteralPath $BuildDirectory -Recurse -Force }
Get-ChildItem -LiteralPath (Join-Path $Work 'src') -Directory -Filter '*.egg-info' -ErrorAction SilentlyContinue | ForEach-Object {
    if (-not $_.FullName.StartsWith($Work, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe package metadata cleanup path.' }
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}
& $EnvironmentPython -m pip install --no-deps --no-build-isolation --force-reinstall $Work
if ($LASTEXITCODE -ne 0) { throw 'Unable to create a self-contained application install for the offline bundle.' }
& $Conda install --prefix $Environment --channel conda-forge --yes conda-pack
if ($LASTEXITCODE -ne 0) { throw 'Unable to install conda-pack in the bundle staging environment.' }
$PackedEnvironment = Join-Path $Work 'runtime-environment.tar.gz'
& (Join-Path $Environment 'Scripts\conda-pack.exe') --prefix $Environment --output $PackedEnvironment --force
if ($LASTEXITCODE -ne 0) { throw 'Unable to create the relocatable runtime environment.' }

$MiniforgeInstall = Join-Path $Work '.runtime\miniforge'
$EnvironmentInstall = Join-Path $Work '.runtime\environment'
if (-not $MiniforgeInstall.StartsWith($Work, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe Miniforge cleanup path.' }
if (-not $EnvironmentInstall.StartsWith($Work, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe environment cleanup path.' }
Remove-Item -LiteralPath $MiniforgeInstall -Recurse -Force
Remove-Item -LiteralPath $EnvironmentInstall -Recurse -Force

$OfflineInstaller = @'
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Spec = Get-Content -LiteralPath (Join-Path $Root 'release-spec.json') -Raw | ConvertFrom-Json
$Installer = Join-Path $Root ('.runtime\downloads\' + $Spec.miniforge.filename)
$Checksum = "$Installer.sha256"
$Expected = ((Get-Content -LiteralPath $Checksum -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
$Actual = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Expected -ne $Actual) { throw 'Bundled Miniforge installer SHA-256 mismatch.' }
$Miniforge = Join-Path $Root '.runtime\miniforge'
$Environment = Join-Path $Root '.runtime\environment'
if (-not (Test-Path -LiteralPath (Join-Path $Miniforge 'python.exe'))) {
    $Process = Start-Process -FilePath $Installer -ArgumentList @('/InstallationType=JustMe','/RegisterPython=0','/S',"/D=$Miniforge") -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "Miniforge installer failed with exit code $($Process.ExitCode)." }
}
if (-not (Test-Path -LiteralPath (Join-Path $Environment 'python.exe'))) {
    New-Item -ItemType Directory -Force -Path $Environment | Out-Null
    & (Join-Path $Miniforge 'python.exe') -m tarfile -e (Join-Path $Root 'runtime-environment.tar.gz') $Environment
    if ($LASTEXITCODE -ne 0) { throw 'Bundled runtime extraction failed.' }
    & (Join-Path $Environment 'Scripts\conda-unpack.exe')
    if ($LASTEXITCODE -ne 0) { throw 'Bundled runtime relocation failed.' }
}
$Python = Join-Path $Root '.runtime\environment\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Bundled runtime is missing.' }
$env:HF_HOME = Join-Path $Root '.runtime-assets\huggingface'
& $Python (Join-Path $Root 'scripts\download_release_models.py') --app-root $Root --hf-home $env:HF_HOME --offline
if ($LASTEXITCODE -ne 0) { throw 'Offline model verification failed.' }
Write-Host 'Offline installation verified. Run .\start.ps1.'
'@
$OfflineInstaller | Set-Content -LiteralPath (Join-Path $Work 'install-offline.ps1') -Encoding UTF8

$Manifest = Get-ChildItem -LiteralPath $Work -File -Recurse | ForEach-Object {
    [pscustomobject]@{
        path = $_.FullName.Substring($Work.Length + 1).Replace('\','/')
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Work 'OFFLINE_MANIFEST.json') -Encoding UTF8
$Zip = Join-Path $OutputDirectory 'osu-ai-detector-v1.0.0-windows-offline.zip'
& "$env:SystemRoot\System32\tar.exe" -a -c -f $Zip -C $OutputDirectory (Split-Path -Leaf $Work)
if ($LASTEXITCODE -ne 0) { throw 'Zip64 offline bundle creation failed.' }
(Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant() | Set-Content -LiteralPath "$Zip.sha256" -Encoding ASCII
Write-Host "Offline bundle created: $Zip"
