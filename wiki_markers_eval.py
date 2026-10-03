"""
wiki_markers_eval.py -- do the markers catalogued in Wikipedia:Signs of AI
writing actually separate machine text from human text *in this corpus*?

That page describes ChatGPT/Grok on encyclopedic prose and says itself that
Claude and Gemini differ. So each marker is treated as a hypothesis:

  1. prevalence   mean rate per 1k words, human vs machine, and the AUC of the
                  single feature on its own (0.5 = no signal; <0.5 = points human)
  2. marginal     does adding it to the existing 18 features move repeated
                  5-fold CV AUC, beyond the noise between seeds?
  3. held-out     train on one generator, test on the other, with and without

Needs corpus/ (see sample_raid.py and gen_modern_par.py).
"""
import json, math, random, re, statistics
import detect
from fit_weights import FEATURES, standardise, fit, auc

# Markers catalogued in "Wikipedia:Signs of AI writing" (CC BY-SA 4.0; credited in
# ATTRIBUTION.md). Phrase lists are observations rather than expression, but the
# credit is theirs. Each is a regex over the apostrophe-folded, lowercased text.
WIKI = {
    "copula_avoid": r"\b(?:serves?|serving|stands?|standing|functions?|operates?)\s+as\s+(?:a|an|the)\b|\b(?:represents|constitutes|marks|boasts)\s+(?:a|an|the)\b",
    "participle_tail": r",\s+(?:highlighting|underscoring|emphasi[sz]ing|ensuring|reflecting|symboli[sz]ing|contributing|cultivating|fostering|encompassing|enhancing|showcasing|demonstrating|illustrating|facilitating|reinforcing|solidifying)\b",
    "significance": r"\btestament\s+to\b|\b(?:pivotal|crucial|vital|key|significant)\s+(?:role|moment|part)\b|\bplays?\s+an?\s+(?:crucial|vital|key|significant|pivotal|important)\s+role\b|\b(?:broader|wider)\s+(?:context|trend|movement|landscape|implications?)\b|\bevolving\s+landscape\b|\bindelible\s+mark\b|\bdeeply\s+rooted\b|\bsetting\s+the\s+stage\b|\b(?:turning|focal)\s+point\b|\b(?:lasting|enduring)\s+(?:legacy|impact|influence)\b|\bunderscor(?:es|ed|ing)\s+(?:the|its)\s+(?:importance|significance)\b",
    "vague_attribution": r"\b(?:experts|observers|critics|analysts|researchers|scholars|commentators)\s+(?:argue|note|believe|suggest|say|agree|have\s+(?:noted|cited|argued|pointed))\b|\bindustry\s+reports?\b|\bsome\s+critics\b|\b(?:widely|generally|broadly)\s+(?:regarded|considered|viewed|seen)\b",
    "promo": r"\bnestled\b|\bin\s+the\s+heart\s+of\b|\brich\s+(?:tapestry|heritage|history|cultural)\b|\bdiverse\s+array\b|\brenowned\b|\bgroundbreaking\b|\bcommitment\s+to\b|\bnatural\s+beauty\b|\bbustling\b|\bstunning\b",
    "vague_connection": r"\b(?:in\s+connection\s+with|in\s+association\s+with|associated\s+with|connected\s+(?:to|with))\b",
    "hedged_availability": r"\bnot\s+(?:widely|extensively|publicly)\s+(?:documented|available|known)\b|\blimited\s+(?:information|data|details)\b|\bbased\s+on\s+(?:the\s+)?available\s+(?:information|sources|data)\b|\bas\s+of\s+my\s+(?:last|latest)\b",
    "chat_residue": r"\bi\s+hope\s+this\s+helps\b|\bof\s+course[!,]|\bcertainly[!,]|\byou'?re\s+absolutely\s+right\b|\bwould\s+you\s+like\b|\bis\s+there\s+anything\s+else\b|\blet\s+me\s+know\b|\bfeel\s+free\s+to\b|\bhere(?:'s|\s+is)\s+(?:a|an|the)\b",
    "outline_conclusion": r"\bdespite\s+(?:its|these|the|this)\b[^.!?]{0,90}?\b(?:faces?|facing)\s+(?:several\s+|some\s+|a\s+number\s+of\s+)?challenges\b|\bfuture\s+(?:outlook|prospects)\b|\bmoving\s+forward\b|\blooking\s+ahead\b",
    "vocab": r"\badditionally\b|\bboasts?\b|\bbolstered\b|\bdeep\s+dive\b|\bemphasi[sz](?:es|ed|ing)\b|\benduring\b|\benhanc(?:e|es|ed|ing)\b|\bgarner(?:s|ed|ing)?\b|\bhighlight(?:s|ed|ing)\b|\binterplay\b|\bintricacies\b|\bshowcas(?:e|es|ed|ing)\b|\bvaluable\b|\balign(?:s|ed|ing)?\s+with\b|\bmeticulously\b|\bunderscor(?:ed|ing)\b",
    "stiff_synonym": r"\b(?:utili[sz](?:e|es|ed|ing)|attempt(?:s|ed|ing)?|relocat(?:e|es|ed|ing)|authored|passed\s+away|commenced|prior\s+to|subsequently|facilitat(?:e|es|ed|ing))\b",
}
_WIKI_RX = {k: re.compile(v) for k, v in WIKI.items()}
_CURLY = re.compile("[‘’“”]")
_COPULA = re.compile(r"\b(?:is|are|was|were)\b")


def extras(text):
    """The Wikipedia-derived features, computed here rather than in detect.py."""
    low = detect.normalise(text).lower()
    n = max(len(detect._words(text)), 1)
    out = {"wiki_%s_per_1k" % k: 1000.0 * len(rx.findall(low)) / n for k, rx in _WIKI_RX.items()}
    out["copula_per_1k"] = 1000.0 * len(_COPULA.findall(low)) / n
    out["curly_quote_per_1k"] = 1000.0 * len(_CURLY.findall(text or "")) / n
    return out


def raw_features(text):
    raw = detect.stylometrics(text)["raw"]
    raw.update(extras(text))
    return raw


NEW = ["wiki_%s_per_1k" % k for k in WIKI] + ["copula_per_1k", "curly_quote_per_1k"]
SEEDS = (7, 11, 13)


def vec(raw, names):
    return [float(raw[n]) if raw.get(n) is not None else 0.45 for n in names]


def cv_auc(raws, y, names, seeds=SEEDS, folds=5):
    X = [vec(r, names) for r in raws]
    Xs, _, _ = standardise(X)
    per_seed = []
    for sd in seeds:
        idx = list(range(len(Xs))); random.Random(sd).shuffle(idx)
        a = []
        for f in range(folds):
            te = set(idx[f::folds])
            tr = [i for i in idx if i not in te]
            w, b = fit([Xs[i] for i in tr], [y[i] for i in tr])
            sc = [sum(w[j] * Xs[i][j] for j in range(len(w))) + b for i in te]
            a.append(auc(sc, [y[i] for i in te]))
        per_seed.append(sum(a) / folds)
    return statistics.mean(per_seed), statistics.pstdev(per_seed)


rows = [json.loads(l) for l in open("corpus/modern-combined.jsonl")]
raws = [raw_features(r["text"]) for r in rows]
y = [r["label"] for r in rows]
H = [r for r, l in zip(raws, y) if l == 0]
M = [r for r, l in zip(raws, y) if l == 1]
print("corpus: %d human / %d machine\n" % (len(H), len(M)))

print("1. PREVALENCE (rate per 1k words) and single-feature AUC")
print("%-34s %8s %8s %7s   %s" % ("feature", "human", "machine", "AUC", "direction"))
for n in NEW + ["hedge_per_1k", "emdash_per_1k", "tier1_per_1k"]:
    h = statistics.mean(vec(r, [n])[0] for r in H)
    m = statistics.mean(vec(r, [n])[0] for r in M)
    a = auc([vec(r, [n])[0] for r in raws], y)
    tag = "machine" if a > 0.55 else ("HUMAN" if a < 0.45 else "~none")
    print("%-34s %8.2f %8.2f %7.3f   %s" % (n, h, m, a, tag))

base, base_sd = cv_auc(raws, y, FEATURES)
print("\n2. MARGINAL  (repeated 5-fold CV over seeds %s; seed-to-seed sd shown)" % (SEEDS,))
print("%-34s %7s %7s %8s" % ("feature set", "AUC", "sd", "vs base"))
print("%-34s %7.4f %7.4f" % ("baseline (18 features)", base, base_sd))
allnew, allnew_sd = cv_auc(raws, y, FEATURES + NEW)
print("%-34s %7.4f %7.4f %+8.4f" % ("baseline + ALL new", allnew, allnew_sd, allnew - base))
for n in NEW:
    a, s = cv_auc(raws, y, FEATURES + [n])
    print("%-34s %7.4f %7.4f %+8.4f" % ("baseline + " + n.replace("_per_1k", ""), a, s, a - base))

print("\n3. HELD-OUT GENERATOR (train on one, test on the other; disjoint humans)")
gem = [r for r in rows if r["model"].startswith("gemma")]
qwn = [r for r in rows if r["model"].startswith("qwen")]
hum = [r for r in rows if r["label"] == 0]
random.Random(5).shuffle(hum)
hA, hB = hum[:len(hum) // 2], hum[len(hum) // 2:]


def logo(train_m, test_m, names):
    def build(machine, humans):
        rs = humans[:len(machine)] + machine
        return [vec(raw_features(r["text"]), names) for r in rs], [r["label"] for r in rs]
    Xtr, ytr = build(train_m, hA); Xte, yte = build(test_m, hB)
    Xs, mean, sd = standardise(Xtr)
    Xt = [[(r[j] - mean[j]) / sd[j] for j in range(len(mean))] for r in Xte]
    w, b = fit(Xs, ytr)
    return auc([sum(w[j] * xi[j] for j in range(len(w))) + b for xi in Xt], yte)


for lab, tr, te in (("Gemma -> Qwen", gem, qwn), ("Qwen -> Gemma", qwn, gem)):
    a0 = logo(tr, te, FEATURES); a1 = logo(tr, te, FEATURES + NEW)
    print("  %-16s baseline %.3f   +new %.3f   (%+.3f)" % (lab, a0, a1, a1 - a0))
