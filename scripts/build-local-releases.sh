#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_DIR="$PROJECT_DIR/releases"
STAGE_DIR="$(mktemp -d /private/tmp/staccato-local-release.XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

cd "$PROJECT_DIR"
npm run build:local

if [ ! -f backend/models/pose_landmarker_lite.task ]; then
  echo "缺少 backend/models/pose_landmarker_lite.task，请先下载姿态识别模型。" >&2
  exit 1
fi

copy_common() {
  local target="$1"
  mkdir -p "$target/backend" "$target/scripts"
  cp backend/__init__.py backend/core.py backend/outpaint.py backend/server.py backend/requirements.txt "$target/backend/"
  mkdir -p "$target/backend/models"
  cp backend/models/pose_landmarker_lite.task "$target/backend/models/"
  mkdir -p "$target/dist/assets"
  cp dist-local/index.html "$target/dist/"
  cp dist-local/assets/index-*.js dist-local/assets/index-*.css "$target/dist/assets/"
  cp docs/本地版_安装与使用说明.md "$target/使用说明.md"
}

MAC_ROOT="$STAGE_DIR/规范切图工作台-macOS"
copy_common "$MAC_ROOT"
cp scripts/local-release/启动规范切图工作台.command "$MAC_ROOT/"
chmod +x "$MAC_ROOT/启动规范切图工作台.command"

WIN_ROOT="$STAGE_DIR/规范切图工作台-Windows"
copy_common "$WIN_ROOT"
cp scripts/local-release/启动规范切图工作台.bat scripts/local-release/停止规范切图工作台.bat "$WIN_ROOT/"
cp scripts/local-release/start-windows.ps1 scripts/local-release/stop-windows.ps1 "$WIN_ROOT/scripts/"

mkdir -p "$RELEASE_DIR"
rm -f "$RELEASE_DIR/staccato-crop-workbench-macos-local-v2.0.zip" "$RELEASE_DIR/staccato-crop-workbench-windows-local-v2.0.zip"
ditto -c -k --keepParent "$MAC_ROOT" "$RELEASE_DIR/staccato-crop-workbench-macos-local-v2.0.zip"
(cd "$STAGE_DIR" && zip -q -r "$RELEASE_DIR/staccato-crop-workbench-windows-local-v2.0.zip" "规范切图工作台-Windows")

echo "已生成："
ls -lh "$RELEASE_DIR/staccato-crop-workbench-macos-local-v2.0.zip" "$RELEASE_DIR/staccato-crop-workbench-windows-local-v2.0.zip"
