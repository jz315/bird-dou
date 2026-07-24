$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$target = Join-Path $repositoryRoot "target\debug\_guandan_native.dll"
$destination = Join-Path $repositoryRoot "python\birddou\_guandan_native.pyd"

cargo build --manifest-path (Join-Path $repositoryRoot "Cargo.toml") -p guandan-pyo3
Copy-Item -LiteralPath $target -Destination $destination -Force
Write-Output "Built $destination"

