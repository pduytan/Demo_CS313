$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (Test-Path ".\.venv\Scripts\python.exe") {
    $py = ".\.venv\Scripts\python.exe"
} elseif (Test-Path "$env:USERPROFILE\miniconda3\envs\cv\python.exe") {
    $py = "$env:USERPROFILE\miniconda3\envs\cv\python.exe"
} else {
    $py = "python"
}

& $py -m streamlit run app.py --server.port 8503 --browser.gatherUsageStats false
