$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $PSScriptRoot
$Port = 8765
$Url = "http://127.0.0.1:$Port"
$RuntimeDir = Join-Path $AppDir ".runtime"
$PidFile = Join-Path $RuntimeDir "server.pid"

try {
  Invoke-RestMethod "$Url/api/health" -TimeoutSec 2 | Out-Null
  Start-Process $Url
  Write-Host "规范切图工作台已经在运行。"
  exit 0
} catch {}

$PythonCommand = $null
$PythonPrefix = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
  foreach ($version in @("-3.12", "-3.11", "-3.10", "-3.9")) {
    & py $version -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $PythonCommand = "py"; $PythonPrefix = @($version); break }
  }
}
if (-not $PythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
  & python -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)"
  if ($LASTEXITCODE -eq 0) { $PythonCommand = "python" }
}
if (-not $PythonCommand) {
  Write-Host "未检测到 Python 3.9–3.12，正在打开官方下载页面。"
  Start-Process "https://www.python.org/downloads/windows/"
  exit 1
}

$VenvPython = Join-Path $AppDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  Write-Host "首次运行：正在创建本地运行环境…"
  & $PythonCommand @PythonPrefix -m venv (Join-Path $AppDir ".venv")
}

$DependencyMarker = Join-Path $AppDir ".venv\.staccato-deps-v2"
if (-not (Test-Path $DependencyMarker)) {
  Write-Host "首次运行：正在安装图像处理组件，通常需要 2–8 分钟…"
  & $VenvPython -m pip install --upgrade pip
  & $VenvPython -m pip install -r (Join-Path $AppDir "backend\requirements.txt")
  if ($LASTEXITCODE -ne 0) { throw "图像处理组件安装失败，请检查网络后重新双击启动。" }
  New-Item -ItemType File -Path $DependencyMarker -Force | Out-Null
}

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
$env:FRONTEND_DIST = Join-Path $AppDir "dist"
$env:PUBLIC_CLOUD = "false"
$env:NO_AI_MODE = "false"
$Stdout = Join-Path $RuntimeDir "server.log"
$Stderr = Join-Path $RuntimeDir "server-error.log"
$Process = Start-Process -FilePath $VenvPython -ArgumentList @("-m", "uvicorn", "backend.server:app", "--host", "127.0.0.1", "--port", "$Port") -WorkingDirectory $AppDir -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -WindowStyle Hidden
$Process.Id | Set-Content -Path $PidFile

for ($i = 0; $i -lt 90; $i++) {
  Start-Sleep -Seconds 1
  if ($Process.HasExited) {
    Write-Host "启动失败："
    if (Test-Path $Stderr) { Get-Content $Stderr -Tail 30 }
    exit 1
  }
  try {
    Invoke-RestMethod "$Url/api/health" -TimeoutSec 2 | Out-Null
    Start-Process $Url
    Write-Host "规范切图工作台已启动：$Url"
    exit 0
  } catch {}
}

Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
throw "启动超时，请查看 .runtime\server-error.log"
