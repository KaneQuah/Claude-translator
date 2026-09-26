"""Stream RAID train.csv -> a balanced, generator-diverse JSONL sample."""
import csv, json, sys
from collections import Counter

csv.field_size_limit(10_000_000)
# RAID's train.csv is ORDERED BY DOMAIN. Quota per (domain, class) and read the
# whole file -- breaking early on a global target yields one domain only.
PER_DOMAIN_HUMAN = 150
PER_DOMAIN_GEN = 20    # per (domain, generator)
OUT = "corpus/raid-sample.jsonl"

human, machine = [], []
gen_counts = Counter()
dom_human = Counter()
seen = set()
rows_read = 0

import os
if not os.path.exists("corpus/train.csv"):
    raise SystemExit(
        "corpus/train.csv is not present (it is 11.8 GB and deliberately not kept).\n"
        "Fetch it first:\n"
        "  hf download liamdugan/raid train.csv --repo-type dataset --local-dir corpus/\n"
        "See ATTRIBUTION.md for why the corpus is not redistributed with this repo.")

with open("corpus/train.csv", newline="", encoding="utf-8", errors="replace") as f:
    for row in csv.DictReader(f):
        rows_read += 1
        if rows_read % 200000 == 0:
            print("  read %d  human=%d machine=%d" % (rows_read, len(human), len(machine)),
                  flush=True)
        if row.get("attack") != "none":
            continue
        g = (row.get("generation") or "").strip()
        if len(g.split()) < 60:
            continue
        key = g[:100]
        if key in seen:
            continue
        model = row.get("model") or "?"
        dom = row.get("domain") or "?"
        if model == "human":
            if dom_human[dom] >= PER_DOMAIN_HUMAN:
                continue
            seen.add(key); dom_human[dom] += 1
            human.append({"text": g, "model": model, "domain": dom, "label": 0})
        else:
            if gen_counts[(dom, model)] >= PER_DOMAIN_GEN:
                continue
            seen.add(key); gen_counts[(dom, model)] += 1
            machine.append({"text": g, "model": model, "domain": dom, "label": 1})

n = min(len(human), len(machine))
import random; random.seed(3)
sample = human[:n] + random.sample(machine, n)
random.shuffle(sample)
with open(OUT, "w") as f:
    for r in sample:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\nrows read: %d" % rows_read)
print("wrote %d (%d human / %d machine) -> %s" % (len(sample), n, n, OUT))
print("generators:", Counter(r["model"] for r in sample).most_common())
print("domains   :", Counter(r["domain"] for r in sample).most_common())
