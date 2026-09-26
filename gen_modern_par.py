"""Parallel version: 4 concurrent requests across the server's slots. Resumes."""
import json, random, re, sys, threading, urllib.request, queue

import os
BASE = os.environ.get("GEN_BASE", "http://127.0.0.1:6773")
MODEL = os.environ.get("GEN_MODEL", "gemma-4-26b-a4b")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
WORKERS = 4
PATH = os.environ.get("GEN_OUT", "corpus/modern-machine.jsonl")

TASK = {
 "abstracts": "Write the abstract of a research paper about: %s",
 "books":     "Write a passage of literary prose that opens: %s",
 "news":      "Write a news article about: %s",
 "poetry":    "Write a poem about: %s",
 "recipes":   "Write a recipe for: %s",
 "reddit":    "Write a Reddit post about: %s",
 "reviews":   "Write a product or film review about: %s",
 "wiki":      "Write a Wikipedia-style article section about: %s",
}

def chat(prompt, max_tokens=300):
    body = json.dumps({"model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9, "top_p": 0.95, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""

rows = [json.loads(l) for l in open("corpus/raid-sample.jsonl")]
humans = [r for r in rows if r["label"] == 0]
random.seed(5); random.shuffle(humans)

try:    done = sum(1 for _ in open(PATH))
except OSError: done = 0
todo = humans[done:N]
print("resuming past %d; generating %d more with %d workers" % (done, len(todo), WORKERS), flush=True)

q = queue.Queue()
for h in todo: q.put(h)
lock = threading.Lock()
f = open(PATH, "a", buffering=1)
count = [0]

def worker():
    while True:
        try: h = q.get_nowait()
        except queue.Empty: return
        try:
            seed = " ".join(h["text"].split()[:18])
            tmpl = TASK.get(h["domain"], "Write a short passage about: %s")
            g = re.sub(r"<think>.*?</think>", "", chat(tmpl % seed), flags=re.S).strip()
            if len(g.split()) >= 60:
                with lock:
                    f.write(json.dumps({"text": g, "model": MODEL,
                                        "domain": h["domain"], "label": 1},
                                       ensure_ascii=False) + "\n")
                    count[0] += 1
                    if count[0] % 20 == 0:
                        print("  +%d (total %d)" % (count[0], done + count[0]), flush=True)
        except Exception as e:
            print("  skip:", str(e)[:60], flush=True)
        finally:
            q.task_done()

ts = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
[t.start() for t in ts]; [t.join() for t in ts]
f.close()
print("done. total machine rows:", sum(1 for _ in open(PATH)))
