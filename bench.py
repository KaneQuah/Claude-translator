"""Benchmark backends on the restyle task: style score, content fidelity, time."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backends, convert, detect, verify


CASES = [
    ("complaint", "the new onboarding docs are useless. they explain the login flow three "
                  "times and never mention how to get a staging key. i had to ask in chat "
                  "like everyone else does."),
    ("nonsense",  "purple mondays fold the toaster sideways. if the ladder sings then "
                  "obviously the carpet owes us nine. never trust a bicycle that remembers "
                  "your birthday."),
    ("technical", "the p100 build needs the sm_60 fp32 patch, it is not upstream. cuda 12.9 "
                  "with driver 580. took 40 minutes to compile on 18 threads and the rpath "
                  "was wrong afterwards."),
    ("question",  "why does the retry loop converge faster when the target score is lower? "
                  "is that just because fewer passes run or is something else going on?"),
]

import os
PASSES = int(os.environ.get("PASSES", "1"))
TARGET = float(os.environ.get("TARGET", "70"))
MODELS = sys.argv[1:] or ["mid", "heavy"]
MODELS = [b for b in MODELS if b in backends.snapshot()["backends"]]
print("benchmarking:", MODELS, flush=True)
rows = []

for backend in MODELS:
    for name, text in CASES:
        t0 = time.time()
        try:
            r = convert.convert(text, intensity="standard", shape="prose",
                                backend=backend, max_attempts=PASSES, target=TARGET,
                                use_judge=False)
            out, err = r["output"], None
        except Exception as exc:
            out, err = "", "%s: %s" % (type(exc).__name__, exc)
        elapsed = time.time() - t0

        row = {"backend": backend, "passes": PASSES, "case": name, "secs": round(elapsed, 1), "error": err}
        if out:
            row["style"] = detect.stylometrics(out)["score"]
            row["in_words"] = len(text.split())
            row["out_words"] = len(out.split())
            v = verify.check(text, out)
            if v.get("available"):
                row.update(added=len(v["added"]), dropped=len(v["dropped"]),
                           contra=len(v["contradicted"]),
                           detail=(v["added"] + v["contradicted"])[:3])
            else:
                row["fidelity_error"] = v.get("reason")
            row["output"] = out
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "output"}), flush=True)

out_path = os.path.join(HERE, "bench-%s-p%d.json" % ("-".join(MODELS), PASSES))
with open(out_path, "w") as f:
    json.dump(rows, f, indent=1, ensure_ascii=False)
print("DONE ->", out_path)
