"""
fit_weights.py -- replace the hand-picked stylometric weights with fitted ones.

Fits a logistic regression on RAID (Dugan et al., ACL 2024) to separate human
from machine-generated text using only the features detect.py already computes.
Pure stdlib: no numpy, so detect.py keeps its zero-dependency property.

The fitted coefficients do two useful things the hand-picked weights could not:
they say how much each feature is actually worth, and their SIGN settles which
direction each feature points -- notably whether low sentence-length variance
really is the machine signal.
"""

import json, math, random, sys
import detect

FEATURES = [
    "tier1_per_1k", "template_per_1k", "assistant_per_1k", "hedge_per_1k",
    "emdash_per_1k", "semicolon_per_1k", "antithesis_per_1k", "tricolon_per_1k",
    "second_person_per_1k", "claudism_per_1k", "paren_per_1k",
    "but_opener_per_1k", "fragment_rate", "sentence_len_cv",
    "sentence_len_mean", "paragraph_len_cv", "bullet_lines", "bold_runs",
]


def featurise(text):
    raw = detect.stylometrics(text)["raw"]
    v = []
    for f in FEATURES:
        x = raw.get(f)
        if x is None:                      # paragraph_len_cv on short texts
            x = 0.45                       # neutral impute
        v.append(float(x))
    return v


def standardise(X):
    n, d = len(X), len(X[0])
    mean = [sum(r[j] for r in X) / n for j in range(d)]
    sd = []
    for j in range(d):
        var = sum((r[j] - mean[j]) ** 2 for r in X) / max(n - 1, 1)
        sd.append(math.sqrt(var) or 1.0)
    return [[(r[j] - mean[j]) / sd[j] for j in range(d)] for r in X], mean, sd


def sigmoid(z):
    if z < -35: return 1e-15
    if z > 35:  return 1 - 1e-15
    return 1.0 / (1.0 + math.exp(-z))


def fit(X, y, iters=3000, lr=0.25, l2=0.02):
    d = len(X[0]); w = [0.0] * d; b = 0.0; n = len(X)
    for _ in range(iters):
        gw = [0.0] * d; gb = 0.0
        for xi, yi in zip(X, y):
            p = sigmoid(sum(w[j] * xi[j] for j in range(d)) + b)
            e = p - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b


def auc(scores, y):
    pairs = sorted(zip(scores, y))
    ranks, i = {}, 0
    # average ranks for ties
    order = [s for s, _ in pairs]
    r = [0.0] * len(pairs)
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and order[j + 1] == order[i]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    pos = sum(1 for _, yy in pairs if yy == 1)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    srp = sum(rr for rr, (_, yy) in zip(r, pairs) if yy == 1)
    return (srp - pos * (pos + 1) / 2.0) / (pos * neg)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus/raid-sample.jsonl"
    rows = [json.loads(l) for l in open(path)]
    print("corpus: %d rows (%d machine / %d human)" % (
        len(rows), sum(r["label"] for r in rows),
        len(rows) - sum(r["label"] for r in rows)))

    X = [featurise(r["text"]) for r in rows]
    y = [r["label"] for r in rows]
    Xs, mean, sd = standardise(X)

    random.seed(7)
    idx = list(range(len(Xs))); random.shuffle(idx)
    folds, aucs, accs = 5, [], []
    for f in range(folds):
        te = set(idx[f::folds])
        Xtr = [Xs[i] for i in range(len(Xs)) if i not in te]
        ytr = [y[i] for i in range(len(y)) if i not in te]
        Xte = [Xs[i] for i in te]; yte = [y[i] for i in te]
        w, b = fit(Xtr, ytr)
        sc = [sum(w[j] * xi[j] for j in range(len(w))) + b for xi in Xte]
        aucs.append(auc(sc, yte))
        accs.append(sum(1 for s, t in zip(sc, yte) if (s > 0) == (t == 1)) / len(yte))
    print("5-fold CV:  AUC %.3f (sd %.3f)   accuracy %.3f" % (
        sum(aucs)/folds, math.sqrt(sum((a-sum(aucs)/folds)**2 for a in aucs)/folds),
        sum(accs)/folds))

    w, b = fit(Xs, y)
    print("\n%-22s %9s %9s   direction" % ("feature", "coef", "weight"))
    tot = sum(abs(x) for x in w) or 1.0
    ranked = sorted(zip(FEATURES, w), key=lambda kv: -abs(kv[1]))
    for name, coef in ranked:
        print("%-22s %9.3f %9.1f   %s" % (
            name, coef, 100 * abs(coef) / tot,
            "higher = machine" if coef > 0 else "higher = HUMAN"))

    json.dump({"features": FEATURES, "coef": w, "intercept": b,
               "mean": mean, "sd": sd,
               "cv_auc": sum(aucs)/folds, "cv_acc": sum(accs)/folds,
               "n": len(rows)},
              open("fitted-weights.json", "w"), indent=1)
    print("\n-> fitted-weights.json")


if __name__ == "__main__":
    main()
