<#
    Offline-friendly Windows build for Synthetic Benchmark Factory.

    Usage (from PowerShell):
      .\build_windows.ps1
      .\build_windows.ps1 -Clean -RunTests
      .\build_windows.ps1 -InstallPyInstaller

    The project has no runtime third-party dependencies. If PyInstaller is
    unavailable, this script still creates a portable source bundle under
    dist\BenchmarkFactory-source and exits successfully with a clear message.
#>
[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$RunTests = $true,
    [switch]$InstallPyInstaller,
    [switch]$NoSourceBundle
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
Set-Location $ProjectRoot

function Find-Python {
    $candidates = @()
    if ($env:PYTHON) { $candidates += $env:PYTHON }
    $candidates += @("py", "python", "python3")
    foreach ($candidate in $candidates) {
        try {
            $null = & $candidate -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch { }
    }
    throw "Python 3 was not found. Install Python 3.10+ and rerun this script."
}

$Python = Find-Python
$dist = Join-Path $ProjectRoot "dist"
$build = Join-Path $ProjectRoot "build"
if ($Clean) {
    foreach ($path in @($dist, $build)) {
        if (Test-Path $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    }
}
New-Item -ItemType Directory -Force -Path $dist | Out-Null

if ($RunTests) {
    Write-Host "[QA] Running standard-library smoke test..." -ForegroundColor Cyan
    & $Python (Join-Path $ProjectRoot "qa_smoke.py")
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed (exit code $LASTEXITCODE)." }
}

$hasPyInstaller = $false
try {
    & $Python -c "import PyInstaller" 2>$null
    $hasPyInstaller = ($LASTEXITCODE -eq 0)
} catch { $hasPyInstaller = $false }

if (-not $hasPyInstaller -and $InstallPyInstaller) {
    Write-Host "[BUILD] Installing optional PyInstaller from configured package indexes..." -ForegroundColor Yellow
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyInstaller." }
    $hasPyInstaller = $true
}

if ($hasPyInstaller) {
    Write-Host "[BUILD] Building dist\BenchmarkFactory.exe..." -ForegroundColor Green
    & $Python -m PyInstaller --noconfirm --clean (Join-Path $ProjectRoot "BenchmarkFactory.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }
    Write-Host "[BUILD] Native executable ready." -ForegroundColor Green
} else {
    Write-Warning "PyInstaller is not installed; producing a portable source bundle instead."
    if (-not $NoSourceBundle) {
        $source = Join-Path $dist "BenchmarkFactory-source"
        if (Test-Path $source) { Remove-Item -LiteralPath $source -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $source | Out-Null
        $include = @("app.py", "synthetic_benchmark_factory", "qa_smoke.py", "README.md", "requirements.txt")
        foreach ($item in $include) {
            $sourcePath = Join-Path $ProjectRoot $item
            if (Test-Path $sourcePath) {
                Copy-Item -LiteralPath $sourcePath -Destination $source -Recurse -Force
            }
        }
        $zip = Join-Path $dist "BenchmarkFactory-source.zip"
        if (Test-Path $zip) { Remove-Item -LiteralPath $zip -Force }
        Compress-Archive -Path (Join-Path $source "*") -DestinationPath $zip -Force
        Write-Host "[BUILD] Portable source bundle: $zip" -ForegroundColor Green
    }
}

Write-Host "[DONE] Synthetic Benchmark Factory build workflow completed." -ForegroundColor Green
