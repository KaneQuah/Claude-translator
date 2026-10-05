"""
server.py -- local web UI for claudeify. Python stdlib only.

    python3 server.py [--port 7080] [--host 127.0.0.1]

Binds to loopback by default. Serves one page and three JSON endpoints; every
model call goes to a llama.cpp server on 127.0.0.1.
"""

import argparse
import json
import os
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import convert
import detect
import verify

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_INPUT = 40000


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("  %s" % (fmt % args))

    # -- helpers ----------------------------------------------------------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_INPUT * 4:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "ui.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/health":
            return self._send(200, {"backends": self._health()})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            data = self._body()
        except Exception as exc:
            return self._send(400, {"error": "bad request: %s" % exc})

        try:
            if path == "/api/convert":
                text = (data.get("text") or "")[:MAX_INPUT]
                if not text.strip():
                    return self._send(400, {"error": "empty input"})
                r = convert.convert(
                    text,
                    intensity=data.get("intensity", "standard"),
                    shape=data.get("shape", "auto"),
                    backend=data.get("backend", "gemma26"),
                    target=(float(data["target"]) if data.get("target") not in (None, "", "auto") else None),
                    max_attempts=int(data.get("max_attempts", 3)),
                    use_judge=bool(data.get("use_judge", True)),
                    disclosure=bool(data.get("disclosure", False)),
                    invisible=bool(data.get("invisible", False)),
                    profile=data.get("profile", "claude"),
                    preserve=bool(data.get("preserve", False)),
                    lexical_boost=bool(data.get("lexical", False)),
                )
                return self._send(200, r)

            if path == "/api/verify":
                orig = (data.get("original") or "")[:MAX_INPUT]
                rew = (data.get("rewrite") or "")[:MAX_INPUT]
                if not orig.strip() or not rew.strip():
                    return self._send(400, {"error": "need both original and rewrite"})
                return self._send(200, verify.check(orig, rew))

            if path == "/api/score":
                text = (data.get("text") or "")[:MAX_INPUT]
                if not text.strip():
                    return self._send(400, {"error": "empty input"})
                return self._send(200, detect.score_text(
                    text, use_model=bool(data.get("use_judge", True)),
                    base=convert.JUDGE["base"], model=convert.JUDGE["model"],
                    profile=data.get("profile", "claude")))

            return self._send(404, {"error": "not found"})

        except Exception as exc:
            traceback.print_exc()
            return self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

    # -- health -----------------------------------------------------------
    @staticmethod
    def _health():
        out = {}
        for name, cfg in convert.BACKENDS.items():
            try:
                with urllib.request.urlopen(cfg["base"] + "/v1/models", timeout=2) as r:
                    json.loads(r.read().decode())
                out[name] = {"up": True, "model": cfg["model"], "base": cfg["base"]}
            except Exception as exc:
                out[name] = {"up": False, "model": cfg["model"],
                             "base": cfg["base"], "error": str(exc)[:120]}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7080)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("claudeify -> http://%s:%d   (ctrl-c to stop)" % (a.host, a.port))
    for name, st in Handler._health().items():
        print("  %-6s %-16s %s" % (name, st["model"], "up" if st["up"] else "DOWN"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
