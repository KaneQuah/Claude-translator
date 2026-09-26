#!/usr/bin/env bash
# Start the claudeify UI. Loopback only.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-7080}"
PID=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [ -n "$PID" ]; then
  echo "port $PORT already in use by pid $PID -- stopping it"
  kill "$PID" 2>/dev/null || true
  sleep 1
fi
exec python3 server.py --port "$PORT"
