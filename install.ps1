[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$SkipBrowserLaunch,
    [string]$ManualModelsDirectory
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Spec = Get-Content -LiteralPath (Join-Path $Root 'release-spec.json') -Raw | ConvertFrom-Json
$RuntimeRoot = Join-Path $Root '.runtime'
$MiniforgeRoot = Join-Path $RuntimeRoot 'miniforge'
$EnvironmentRoot = Join-Path $RuntimeRoot 'environment'
$DownloadRoot = Join-Path $RuntimeRoot 'downloads'
$PackageCacheRoot = Join-Path $RuntimeRoot 'conda-pkgs'
$PipCacheRoot = Join-Path $RuntimeRoot 'pip-cache'
$Installer = Join-Path $DownloadRoot $Spec.miniforge.filename
$ChecksumFile = "$Installer.sha256"
$CondaConfig = Join-Path $RuntimeRoot 'condarc-release.yml'
$EnvironmentSpec = Join-Path $Root 'environment-release.yml'
$EnvironmentMarker = Join-Path $EnvironmentRoot '.release-environment-complete'
$EnvironmentSpecHash = (Get-FileHash -LiteralPath $EnvironmentSpec -Algorithm SHA256).Hash.ToLowerInvariant()
$ManualModelsRoot = if ([string]::IsNullOrWhiteSpace($ManualModelsDirectory)) {
    Join-Path $Root 'manual-models'
} elseif ([System.IO.Path]::IsPathRooted($ManualModelsDirectory)) {
    [System.IO.Path]::GetFullPath($ManualModelsDirectory)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $Root $ManualModelsDirectory))
}

function Show-ManualModelInstructions {
    Write-Host @"

Automatic model download did not complete.

You can download the missing repositories with a browser and place their files
in these folders (completed repositories already present in the cache may be
omitted):

  $ManualModelsRoot\osu-ai-detector-models
  $ManualModelsRoot\v29
  $ManualModelsRoot\v30
  $ManualModelsRoot\v31
  $ManualModelsRoot\v32
  $ManualModelsRoot\v32-mini

Download pages and pinned revisions:
  https://huggingface.co/NettoAndTetto/osu-ai-detector-models/tree/v1.0.0
  https://huggingface.co/OliBomby/Mapperatorinator-v29.1/tree/656db0cd04a8a6a77d94a96e7af89810fb6de5ef
  https://huggingface.co/OliBomby/Mapperatorinator-v30/tree/a4c6e6e69c055711c2293d63161c0e52980e56a1
  https://huggingface.co/OliBomby/Mapperatorinator-v31/tree/12772791b862b97a11153aa766b2481afa5dda11
  https://huggingface.co/OliBomby/Mapperatorinator-v32/tree/74f22583400d259bf424819e11027c17933efe54
  https://huggingface.co/OliBomby/Mapperatorinator-v32-mini/tree/7807f0dc70cab671be012e1f5ddf945b0b8b7278

Preserve each repository's directory structure. For v29-v31, the required
files are at the folder root. For v32 and v32-mini, they are under gamemode=0.
The installer checks every required file's byte size and SHA-256. Then rerun:

  .\install.ps1 -ManualModelsDirectory "$ManualModelsRoot"

The installer will verify and import the local files. It will not redownload
repositories already available in the pinned cache.
"@
}

function Remove-ProcessEnvironmentVariable {
    param([Parameter(Mandatory = $true)][string]$Name)
    Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
}

function Remove-PartialApplicationEnvironment {
    if (-not (Test-Path -LiteralPath $EnvironmentRoot)) { return }
    $ResolvedRuntime = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\')
    $ResolvedEnvironment = [System.IO.Path]::GetFullPath($EnvironmentRoot).TrimEnd('\')
    $ExpectedEnvironment = [System.IO.Path]::Combine($ResolvedRuntime, 'environment')
    if (-not $ResolvedEnvironment.Equals($ExpectedEnvironment, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove unexpected environment path: $ResolvedEnvironment"
    }
    Write-Host 'Removing the incomplete application environment; downloaded package cache will be retained.'
    Remove-Item -LiteralPath $ResolvedEnvironment -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $DownloadRoot | Out-Null
if (-not (Test-Path -LiteralPath $Installer) -or -not (Test-Path -LiteralPath $ChecksumFile)) {
    Write-Host 'Downloading missing pinned Miniforge files...'
    try {
        if (-not (Test-Path -LiteralPath $Installer)) {
            Invoke-WebRequest -UseBasicParsing -Uri $Spec.miniforge.url -OutFile $Installer
        }
        if (-not (Test-Path -LiteralPath $ChecksumFile)) {
            Invoke-WebRequest -UseBasicParsing -Uri $Spec.miniforge.sha256_url -OutFile $ChecksumFile
        }
    } catch {
        Write-Host "Download the installer manually to: $Installer"
        Write-Host "Installer URL: $($Spec.miniforge.url)"
        Write-Host "Download its checksum to: $ChecksumFile"
        Write-Host "Checksum URL: $($Spec.miniforge.sha256_url)"
        throw 'Miniforge download failed. Place both files at the paths above and rerun install.ps1.'
    }
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
$EnvironmentReady = (
    (Test-Path -LiteralPath (Join-Path $EnvironmentRoot 'python.exe')) -and
    (Test-Path -LiteralPath $EnvironmentMarker) -and
    ((Get-Content -LiteralPath $EnvironmentMarker -Raw).Trim() -eq $EnvironmentSpecHash)
)
if (-not $EnvironmentReady) {
    # The public installer must not inherit an activated Anaconda/Conda runtime.
    # Keep ordinary proxy and corporate CA settings, but remove certificates that
    # point inside the caller's active Conda installation.
    $ExternalCondaPrefix = $env:CONDA_PREFIX
    foreach ($CertificateVariable in @('SSL_CERT_FILE','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE')) {
        $CertificatePath = [Environment]::GetEnvironmentVariable($CertificateVariable, 'Process')
        if ($ExternalCondaPrefix -and $CertificatePath) {
            try {
                $ResolvedCertificate = [System.IO.Path]::GetFullPath($CertificatePath)
                $ResolvedExternalPrefix = [System.IO.Path]::GetFullPath($ExternalCondaPrefix).TrimEnd('\') + '\'
                if ($ResolvedCertificate.StartsWith($ResolvedExternalPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                    Remove-ProcessEnvironmentVariable $CertificateVariable
                }
            } catch {
                # Leave unrelated or non-file certificate configuration untouched.
            }
        }
    }
    foreach ($Variable in @(
        'CONDA_PREFIX','CONDA_DEFAULT_ENV','CONDA_EXE','_CONDA_EXE','_CONDA_ROOT',
        'CONDA_PYTHON_EXE','CONDA_PROMPT_MODIFIER','CONDA_SHLVL','_CE_CONDA','_CE_M'
    )) {
        Remove-ProcessEnvironmentVariable $Variable
    }

    @'
channels:
  - conda-forge
channel_priority: strict
show_channel_urls: true
'@ | Set-Content -LiteralPath $CondaConfig -Encoding ASCII
    $env:CONDARC = $CondaConfig
    $env:CONDA_CHANNEL_PRIORITY = 'strict'
    New-Item -ItemType Directory -Force -Path $PackageCacheRoot, $PipCacheRoot | Out-Null
    $env:CONDA_PKGS_DIRS = $PackageCacheRoot
    $env:PIP_CACHE_DIR = $PipCacheRoot
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONSAFEPATH = '1'

    Write-Host 'Creating the pinned application environment...'
    Remove-PartialApplicationEnvironment
    & $Conda env create --prefix $EnvironmentRoot --file $EnvironmentSpec --yes --no-default-packages
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $EnvironmentRoot 'python.exe'))) {
        throw "Conda environment creation failed. Rerun once after checking network/proxy access, or build the documented offline bundle on another connected Windows machine. Downloaded packages remain cached at $PackageCacheRoot."
    }
    $EnvironmentSpecHash | Set-Content -LiteralPath $EnvironmentMarker -Encoding ASCII
}

$Python = Join-Path $EnvironmentRoot 'python.exe'
$env:HF_HOME = Join-Path $Root '.runtime-assets\huggingface'
$env:HF_HUB_ETAG_TIMEOUT = '30'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '180'
$env:HF_HUB_DISABLE_XET = '1'
if (Test-Path -LiteralPath $ManualModelsRoot -PathType Container) {
    Write-Host "Importing and verifying manually downloaded model assets from $ManualModelsRoot ..."
    & $Python (Join-Path $Root 'scripts\download_release_models.py') --app-root $Root --hf-home $env:HF_HOME --manual-root $ManualModelsRoot --offline
    if ($LASTEXITCODE -ne 0) {
        Show-ManualModelInstructions
        throw 'Manual model import failed. Complete the indicated folders and rerun install.ps1.'
    }
} else {
    Write-Host 'Downloading and verifying calibrated and upstream model assets (one automatic attempt)...'
    & $Python (Join-Path $Root 'scripts\download_release_models.py') --app-root $Root --hf-home $env:HF_HOME
    if ($LASTEXITCODE -ne 0) {
        Show-ManualModelInstructions
        throw 'Automatic model download failed. Use the manual-models fallback above and rerun install.ps1.'
    }
}

$Vendor = Join-Path $Root 'vendor\Mapperatorinator'
$VendorRevision = Join-Path $Vendor '.release-source-revision'
if (-not (Test-Path -LiteralPath $VendorRevision)) {
    throw 'The release is missing its bundled Mapperatorinator source revision marker. Download a fresh release archive.'
}
if ((Get-Content -LiteralPath $VendorRevision -Raw).Trim() -ne $Spec.upstream_source.revision) {
    throw 'Bundled Mapperatorinator source revision does not match release-spec.json.'
}

$Receipt = @{
    schema_version = 1
    release = $Spec.tag
    installed_at_utc = [DateTime]::UtcNow.ToString('o')
    miniforge_sha256 = $Actual
    cpu_only_requested = [bool]$CpuOnly
} | ConvertTo-Json
$Receipt | Set-Content -LiteralPath (Join-Path $RuntimeRoot 'installation-receipt.json') -Encoding UTF8
Write-Host 'Installation complete. Run .\start.ps1 to launch the local application.'
