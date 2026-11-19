#!/bin/bash
# 容器入口脚本：同时启动 FastAPI 后端与 Streamlit 前端
# - 两个进程都在同一容器内，通过 127.0.0.1 通信
# - 任一进程退出时关闭另一个并结束容器（配合 restart 策略可自动重启）
set -e

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"

echo "[entrypoint] 启动 FastAPI 后端  http://${BACKEND_HOST}:${BACKEND_PORT}"
uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!

echo "[entrypoint] 启动 Streamlit 前端 http://0.0.0.0:${FRONTEND_PORT}"
streamlit run frontend/app.py \
  --server.port "$FRONTEND_PORT" \
  --server.address 0.0.0.0 \
  --server.headless true &
FRONTEND_PID=$!

shutdown() {
  trap '' TERM INT
  echo "[entrypoint] 正在停止服务..."
  kill -TERM "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap shutdown TERM INT

# 任一进程退出即结束容器（配合 restart 策略自动重启）
EXIT_CODE=0
wait -n || EXIT_CODE=$?
shutdown
exit "$EXIT_CODE"
