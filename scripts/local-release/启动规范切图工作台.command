#!/bin/bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"
PORT=8765
URL="http://127.0.0.1:${PORT}"

if curl -fsS "${URL}/api/health" >/dev/null 2>&1; then
  open "$URL"
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "未检测到 Python 3。请先安装 Python 3.9–3.12：https://www.python.org/downloads/macos/"
  read -r -p "按回车键打开下载页面…"
  open "https://www.python.org/downloads/macos/"
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)'; then
  echo "需要 Python 3.9–3.12。当前版本：$(python3 --version 2>&1)"
  read -r -p "按回车键打开 Python 下载页面…"
  open "https://www.python.org/downloads/macos/"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "首次运行：正在创建本地运行环境…"
  python3 -m venv .venv
fi

if [ ! -f ".venv/.staccato-deps-v2" ]; then
  echo "首次运行：正在安装图像处理组件，通常需要 2–8 分钟…"
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r backend/requirements.txt
  touch .venv/.staccato-deps-v2
fi

mkdir -p .runtime
export FRONTEND_DIST="$APP_DIR/dist"
export PUBLIC_CLOUD=false
export NO_AI_MODE=false
.venv/bin/python -m uvicorn backend.server:app --host 127.0.0.1 --port "$PORT" >.runtime/server.log 2>.runtime/server-error.log &
SERVER_PID=$!
echo "$SERVER_PID" > .runtime/server.pid

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  rm -f .runtime/server.pid
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 90); do
  if curl -fsS "${URL}/api/health" >/dev/null 2>&1; then
    open "$URL"
    echo "规范切图工作台已启动：$URL"
    echo "请保留此窗口；关闭窗口或按 Control+C 即可停止工具。"
    wait "$SERVER_PID"
    exit $?
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "启动失败，请查看：$APP_DIR/.runtime/server-error.log"
    tail -30 .runtime/server-error.log || true
    read -r -p "按回车键关闭…"
    exit 1
  fi
  sleep 1
done

echo "启动超时，请查看：$APP_DIR/.runtime/server-error.log"
exit 1
