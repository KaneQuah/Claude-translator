#!/usr/bin/env bash
# Start the claudeify UI on all interfaces; server.py drops any client outside
# its --allow list (loopback, Tailscale, private LAN). HOST=127.0.0.1 to stay local.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-7080}"
PID=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [ -n "$PID" ]; then
  echo "port $PORT already in use by pid $PID -- stopping it"
  kill "$PID" 2>/dev/null || true
  sleep 1
fi
exec python3 server.py --host "${HOST:-0.0.0.0}" --port "$PORT"
