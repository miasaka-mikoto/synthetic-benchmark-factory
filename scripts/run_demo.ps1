param(
  [int]$Count = 500,
  [int]$Seed = 20261004,
  [string]$Out = "artifacts/demo"
)

$ErrorActionPreference = "Stop"
python run_demo.py --count $Count --seed $Seed --out $Out
if ($LASTEXITCODE -ne 0) { throw "Synthetic dataset validation failed" }
Write-Host "Demo dataset written to $Out"

