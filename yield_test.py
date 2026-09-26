"""
Is rejection-sampling -> LoRA viable, or is prompting already at the ceiling?

Generates N candidates per input at high temperature and counts how many clear
BOTH bars at once: heavy style marking AND zero invented content. That fraction
is the yield of an SFT corpus built by rejection sampling -- if it is decent,
fine-tuning is cheap; if it is near zero, prompting is not the bottleneck, the
objective is.
"""
import json, sys, time
sys.path.insert(0, "/home/kane/ai-stack/claudeify")
import convert, detect, verify

INPUTS_SHORT = [
 "the new onboarding docs are useless. they explain the login flow three times and never mention how to get a staging key.",
 "purple mondays fold the toaster sideways. if the ladder sings then obviously the carpet owes us nine.",
 "the p100 build needs the sm_60 fp32 patch, it is not upstream. cuda 12.9 with driver 580. took 40 minutes on 18 threads.",
 "why does the retry loop converge faster when the target score is lower?",
 "standup is a waste of time. nobody remembers yesterday and we just read the board out loud.",
 "the cache never invalidates on schema change. we found out in production, twice.",
 "i think the join is duplicating rows but i cannot prove it yet.",
 "we should pin the deps properly or this keeps happening every few weeks.",
]

# Longer inputs: does the style ceiling for faithful output lift when the source
# actually supplies enough material to build tricolons and contrasts from?
INPUTS_LONG = [
 "the onboarding docs are useless. they explain the login flow three separate times, once in the overview, once in the quickstart and once in the appendix, but they never mention how to get a staging key. i asked in chat and three other people said they had hit the same wall that week. the wiki page that used to cover it was deleted in march and nobody replaced it.",
 "the p100 build needs the sm_60 fp32 patch and it is not upstream, only sm_61 is. we are on cuda 12.9 with driver 580, which is the last branch that supports pascal. the compile took forty minutes on eighteen threads and afterwards the rpath pointed at the wrong cuda directory, so the binary loaded the system libs instead of ours and segfaulted on the first kernel launch.",
 "standup is a waste of time. nobody remembers what they said yesterday, we read the ticket board out loud, and the two people who actually need to sync end up talking again afterwards anyway. it costs eight people fifteen minutes every morning. i would rather we posted written updates in a channel and kept the meeting for when something is actually blocked.",
 "the cache never invalidates on schema change. we found out in production twice, once when the price field changed type and once when we added a nullable column. both times the stale entries served for about six hours before anyone noticed. the fix is probably to key the cache on a schema hash but nobody has had time to do it properly.",
 "we tried three approaches before settling on this one. the first was too slow on large inputs, the second broke whenever the upstream schema changed which happened twice in a month, and the third is what we have now. it is not elegant but it has run without intervention since march. at this point that counts for more than elegance does.",
 "the retry loop works but i am not sure the scoring weights are right. i picked them by hand from a list of markers and tuned the ramps by eye against maybe five texts. it might be measuring nothing at all. what worries me is that it looks rigorous, so nobody will question the number, and it will end up in a report somewhere as though it means something.",
 "the deploy script silently skips the migration step when the env var is unset. it took two days to find because the logs show a successful deploy either way. there is no warning, no exit code, nothing. it should fail loudly instead, or at minimum print which steps it decided to skip and why.",
 "our test suite has a hundred and sixty nine passing tests and seven of the eight real input files break it. the tests were written against inputs the original developer made up, so they all share the same shape. the suite being green tells you the code handles that one shape, and nothing else. it is worse than no tests because it creates confidence.",
]
import os
INPUTS = INPUTS_LONG if os.environ.get("LONG") else INPUTS_SHORT
print("input set: %s (mean %d words)" % (
    "LONG" if os.environ.get("LONG") else "SHORT",
    sum(len(t.split()) for t in INPUTS)//len(INPUTS)), flush=True)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
STYLE_BAR = 70.0

rows = []
for text in INPUTS:
    for k in range(K):
        t0 = time.time()
        try:
            r = convert.convert(text, shape="prose", backend="gemma26",
                                max_attempts=1, use_judge=False, temperature=1.0)
        except Exception as e:
            print("gen error:", str(e)[:80], flush=True); continue
        out = r["output"]
        sty = r["report"]["stylometric"]["style_density"]
        v = verify.check(text, out)
        added = len(v.get("added", [])) + len(v.get("contradicted", []))
        dropped = len(v.get("dropped", []))
        ok = (sty >= STYLE_BAR) and added == 0 and dropped == 0
        rows.append({"input": text[:40], "style": sty, "added": added,
                     "dropped": dropped, "keep": ok, "words": len(out.split()),
                     "secs": round(time.time()-t0, 1), "output": out})
        print("  style=%5.1f added=%d dropped=%d %s" % (sty, added, dropped,
              "KEEP" if ok else ""), flush=True)

n = len(rows)
keep = sum(1 for r in rows if r["keep"])
hi = sum(1 for r in rows if r["style"] >= STYLE_BAR)
clean = sum(1 for r in rows if r["added"] == 0 and r["dropped"] == 0)
print("\ncandidates      : %d" % n)
print("style >= %.0f     : %d (%.0f%%)" % (STYLE_BAR, hi, 100.0*hi/n))
print("zero drift      : %d (%.0f%%)" % (clean, 100.0*clean/n))
print("BOTH (SFT yield): %d (%.0f%%)" % (keep, 100.0*keep/n))
import os as _os
json.dump(rows, open("yield-results%s.json" % ("-long" if _os.environ.get("LONG") else ""), "w"), indent=1, ensure_ascii=False)
print("-> yield-results.json")
