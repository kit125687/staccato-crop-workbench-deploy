$AppDir = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $AppDir ".runtime\server.pid"
if (Test-Path $PidFile) {
  $ServerPid = Get-Content $PidFile | Select-Object -First 1
  if ($ServerPid) { Stop-Process -Id ([int]$ServerPid) -Force -ErrorAction SilentlyContinue }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  Write-Host "规范切图工作台已停止。"
} else {
  Write-Host "未发现正在运行的规范切图工作台。"
}
