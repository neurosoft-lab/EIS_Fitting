param([Parameter(ValueFromRemainingArguments=$true)][string[]]$AppArguments)
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$entry = Join-Path $PSScriptRoot 'launch.pyw'
$candidates = [System.Collections.Generic.List[string]]::new()
if ($env:EIS_PYTHON) { $candidates.Add($env:EIS_PYTHON) }
$candidates.Add((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'))
$store = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps'
Get-ChildItem -LiteralPath $store -Directory -Filter 'PythonSoftwareFoundation.Python.*' -ErrorAction SilentlyContinue | ForEach-Object {
    $candidates.Add((Join-Path $_.FullName 'python.exe'))
}
foreach ($name in @('python.exe','python3.exe')) {
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($command) { $candidates.Add($command.Source) }
}
$py = Get-Command py.exe -ErrorAction SilentlyContinue
if ($py) {
    $ErrorActionPreference = 'Continue'
    $installed = & $py.Source -0p 2>&1
    $ErrorActionPreference = 'Stop'
    $installed | ForEach-Object {
        if ("$_" -match '([A-Za-z]:\\.*python.exe)') { $candidates.Add($Matches[1]) }
    }
}
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
    try {
        & $candidate -B $entry --check 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { continue }
    } catch { continue }
    if ($AppArguments.Count -gt 0) {
        $ErrorActionPreference = 'Continue'
        & $candidate -B $entry @AppArguments
        exit $LASTEXITCODE
    }
    $windowless = Join-Path (Split-Path $candidate) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $windowless)) { $windowless = $candidate }
    Start-Process -FilePath $windowless -ArgumentList @('-B',('"' + $entry + '"')) -WindowStyle Hidden
    exit 0
}
Write-Host 'A compatible Python environment was not found. See 03_Bin/README.md.'
Read-Host 'Press Enter to close'
exit 1
