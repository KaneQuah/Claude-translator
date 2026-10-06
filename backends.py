"""
backends.py -- which models can claudeify talk to?

The app never scans your disk. It asks the servers you point it at, over the
OpenAI-compatible `GET /v1/models` that Ollama, llama.cpp's llama-server, LM Studio
and vLLM all expose, and offers whatever they list.

Where it looks, in order:

  1. `backends.json` next to this file, if present (git-ignored; see
     `backends.example.json`). Named entries with a fixed base URL and model,
     plus optional "default", "judge" and "auditor".
  2. `CLAUDEIFY_ENDPOINT`: comma-separated server URLs. Default is the two usual
     local ports: Ollama (11434) and llama-server (8080).

Other environment variables:
  CLAUDEIFY_MODEL   model id to preselect
  CLAUDEIFY_JUDGE   llama.cpp server URL for the perplexity pass. It must be
                    llama.cpp: the pass needs /tokenize and per-token probabilities
                    on /completion, which Ollama's API does not offer. If unset, a
                    llama-server on :8080 is picked up automatically.

Stdlib only.
"""

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENDPOINTS = "http://127.0.0.1:11434,http://127.0.0.1:8080"
PROBE_TIMEOUT = 4
CACHE_TTL = 10

_cache = {"t": 0.0, "v": None}


def _config():
    try:
        with open(os.path.join(HERE, "backends.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def norm(base):
    """'http://host:11434/v1/' -> 'http://host:11434' (we add /v1 ourselves)."""
    base = base.strip().rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def endpoints():
    spec = os.environ.get("CLAUDEIFY_ENDPOINT", DEFAULT_ENDPOINTS)
    return [norm(x) for x in spec.split(",") if x.strip()]


def _list(base):
    """Ask one server what it serves. Returns (state, [model ids]).

    state: 'up', 'busy' (alive but too slow to answer, e.g. mid-generation on a
    single slot) or 'down' (refused / unreachable). A busy server stays usable.
    """
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=PROBE_TIMEOUT) as r:
            d = json.loads(r.read().decode())
        return "up", [m["id"] for m in d.get("data", []) if m.get("id")]
    except Exception as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason):
            return "busy", []
        return "down", []


def _is_embedding(model_id):
    return bool(re.search(r"embed|rerank|bge-|nomic", model_id, re.I))


def snapshot(force=False):
    """Everything the UI needs: models, which one to preselect, judge, endpoints."""
    if not force and _cache["v"] and time.time() - _cache["t"] < CACHE_TTL:
        return _cache["v"]
    cfg = _config()
    named = cfg.get("backends", {})
    eps = [] if named else endpoints()

    with ThreadPoolExecutor(max_workers=max(1, len(named) + len(eps))) as ex:
        named_f = {n: ex.submit(_list, b["base"]) for n, b in named.items()}
        eps_f = {e: ex.submit(_list, e) for e in eps}
        named_r = {n: f.result() for n, f in named_f.items()}
        eps_r = {e: f.result() for e, f in eps_f.items()}

    models = {}
    for n, b in named.items():
        state = named_r[n][0]
        models[n] = {"base": norm(b["base"]), "model": b["model"],
                     "up": state != "down", "busy": state == "busy"}
    ep_info = []
    for e, (state, ids) in eps_r.items():
        usable = [i for i in ids if not _is_embedding(i)]
        ep_info.append({"base": e, "state": state, "models": len(usable)})
        for i in usable:
            name = i if i not in models else "%s @ %s" % (i, e)
            models[name] = {"base": e, "model": i, "up": True, "busy": False}

    snap = {"backends": models, "endpoints": ep_info,
            "default": _pick_default(models, cfg), "judge": judge(cfg, eps_r)}
    _cache.update(t=time.time(), v=snap)
    return snap


def _pick_default(models, cfg):
    live = [n for n, m in models.items() if m["up"]]
    want = os.environ.get("CLAUDEIFY_MODEL") or cfg.get("default")
    if want and want in models:
        return want
    for n in live:                      # measured best: Gemma 4 26B-A4B
        if re.search(r"gemma[-_ ]?4.*26", models[n]["model"], re.I):
            return n
    return live[0] if live else None


LLAMACPP_DEFAULT = "http://127.0.0.1:8080"


def _has_tokenize(base):
    """Only llama.cpp's server has POST /tokenize (Ollama and the OpenAI API do not)."""
    try:
        req = urllib.request.Request(base + "/tokenize", json.dumps({"content": "hi"}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as r:
            return "tokens" in json.loads(r.read().decode())
    except Exception:
        return False


def judge(cfg=None, probed=None):
    """The perplexity pass needs per-token probabilities for text the server did not
    write, which only llama.cpp's /tokenize + /completion can give. So it is pinned
    to a llama.cpp server regardless of which backend does the restyling: configured
    explicitly, or auto-detected on llama-server's default port.
    A small model other than the restyler is best: a model judging its own output
    finds it unusually predictable."""
    cfg = cfg if cfg is not None else _config()
    if cfg.get("judge"):
        j = cfg["judge"]
        return {"base": norm(j["base"]), "model": j.get("model", "judge")}
    if os.environ.get("CLAUDEIFY_JUDGE"):
        return {"base": norm(os.environ["CLAUDEIFY_JUDGE"]), "model": "judge"}
    state = (probed or {}).get(LLAMACPP_DEFAULT, ("down", []))[0]
    if state == "up" and _has_tokenize(LLAMACPP_DEFAULT):
        return {"base": LLAMACPP_DEFAULT, "model": "judge"}
    return None


def auditor(cfg=None):
    cfg = cfg if cfg is not None else _config()
    a = cfg.get("auditor")
    return {"base": norm(a["base"]), "model": a["model"]} if a else None


def resolve(name=None):
    """Name (or None for the default) -> {'base', 'model'}. Raises if nothing is up."""
    snap = snapshot()
    models = snap["backends"]
    if name in models:
        return models[name]
    if name:                            # a bare model id also works
        for m in models.values():
            if m["model"] == name:
                return m
    d = snap["default"]
    if d:
        return models[d]
    raise RuntimeError(no_models_message(snap))


def no_models_message(snap=None):
    snap = snap or snapshot()
    where = ", ".join(e["base"] for e in snap["endpoints"]) or "backends.json"
    return ("No models found at %s. Start Ollama and pull one (ollama pull gemma4:26b), "
            "or set CLAUDEIFY_ENDPOINT to your server's URL." % where)


# Thinking models spend the whole budget on reasoning unless told not to, and the
# switch differs by server: llama.cpp reads chat_template_kwargs, Ollama reads
# reasoning_effort. Send both; a server that rejects unknown fields gets a plain retry.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False},
               "reasoning_effort": "none"}


def post_chat(payload, base, timeout=300):
    for body in (dict(payload, **NO_THINKING), payload):
        req = urllib.request.Request(
            base + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code != 400 or body is payload:
                raise
    raise RuntimeError("unreachable")
