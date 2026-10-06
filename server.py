"""
server.py -- local web UI for claudeify. Python stdlib only.

    python3 server.py [--port 7080] [--host 127.0.0.1] [--allow CIDR,...]

Binds to loopback by default; run.sh binds 0.0.0.0 so a Tailscale or LAN device
can reach it. Serves one page and three JSON endpoints; every model call goes to
a llama.cpp server on 127.0.0.1, which is never exposed.

There is no login, so reachability is the access control, in three layers:
  1. client address must fall in --allow (default: loopback, Tailscale
     100.64.0.0/10, and the private LAN ranges) -- anything else is dropped;
  2. the Host header must be an allowed IP, localhost, this machine's name or a
     *.ts.net name, which defeats DNS rebinding from a hostile web page;
  3. POSTs must be application/json and, if they carry an Origin, same-origin.
At most MAX_BUSY requests hit the GPU at once; the rest get a 429.
"""

import argparse
import ipaddress
import json
import os
import socket
import threading
import traceback
import urllib.request
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import convert
import detect
import verify

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_INPUT = 40000
MAX_BUSY = 2
DEFAULT_ALLOW = ("127.0.0.0/8,100.64.0.0/10,10.0.0.0/8,172.16.0.0/12,"
                 "192.168.0.0/16")
ALLOW = []
BUSY = threading.BoundedSemaphore(MAX_BUSY)
SELF_NAMES = {"localhost", socket.gethostname().lower()}
CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
       "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; "
       "frame-ancestors 'none'")


def parse_allow(spec):
    return [ipaddress.ip_network(x.strip(), strict=False)
            for x in spec.split(",") if x.strip()]


def ip_allowed(addr):
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return any(ip in net for net in ALLOW)


def host_allowed(host_header):
    """Host header is an allowed IP literal, this machine, or a tailnet name."""
    if not host_header:
        return False
    h = urlsplit("//" + host_header).hostname or ""
    h = h.lower()
    if h in SELF_NAMES or h.endswith(".ts.net"):
        return True
    return ip_allowed(h)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def verify_request(self, request, client_address):
        return ip_allowed(client_address[0])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30        # idle socket reads only; a slow client cannot pin a thread

    def log_message(self, fmt, *args):
        print("  %s  %s" % (self.client_address[0], fmt % args))

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
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if ctype.startswith("text/html"):
            self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_INPUT * 4:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    def _reject(self, post=False):
        """Return True (after replying) if the request fails the Host/Origin checks."""
        host = self.headers.get("Host", "")
        if not host_allowed(host):
            self._send(421, {"error": "unrecognised Host header"})
            return True
        if post:
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != host:
                self._send(403, {"error": "cross-origin request refused"})
                return True
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._send(415, {"error": "application/json required"})
                return True
        return False

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        if self._reject():
            return
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "ui.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/health":
            return self._send(200, {"backends": self._health()})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self._reject(post=True):
            return
        path = self.path.split("?")[0]
        try:
            data = self._body()
        except Exception as exc:
            return self._send(400, {"error": "bad request: %s" % exc})

        if path not in ("/api/convert", "/api/verify", "/api/score"):
            return self._send(404, {"error": "not found"})
        if not BUSY.acquire(blocking=False):
            return self._send(429, {"error": "busy: %d requests already running, try again shortly" % MAX_BUSY})
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
            return self._send(500, {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
        finally:
            BUSY.release()

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
    ap.add_argument("--allow", default=DEFAULT_ALLOW,
                    help="comma-separated CIDRs allowed to connect")
    a = ap.parse_args()
    ALLOW[:] = parse_allow(a.allow)
    srv = Server((a.host, a.port), Handler)
    print("claudeify -> http://%s:%d   allow: %s   (ctrl-c to stop)"
          % (a.host, a.port, a.allow))
    for name, st in Handler._health().items():
        print("  %-6s %-16s %s" % (name, st["model"], "up" if st["up"] else "DOWN"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
