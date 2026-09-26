"""
Leave-one-generator-out: train on one modern generator, test on the other.

A fit trained on the same model the tool uses could be learning that model's
quirks rather than machine-ness in general. RAID's whole thesis is that this is
where detectors quietly fail, so the honest check is to hold a generator out.
"""
import json, math, random, sys
import detect
from fit_weights import FEATURES, featurise, standardise, fit, auc

human = [json.loads(l) for l in open("corpus/raid-sample.jsonl") if json.loads(l)["label"] == 0]
gem = [json.loads(l) for l in open("corpus/modern-machine.jsonl")]
qwn = [json.loads(l) for l in open("corpus/modern-machine-qwen.jsonl")]
random.seed(5); random.shuffle(human)

print("human %d   gemma %d   qwen %d" % (len(human), len(gem), len(qwn)))

def build(machine, humans):
    rows = humans[:len(machine)] + machine
    return [featurise(r["text"]) for r in rows], [r["label"] for r in rows]

# Disjoint human pools so the held-out set shares no human text with training.
half = len(human) // 2
hA, hB = human[:half], human[half:]

def run(train_m, train_h, test_m, test_h, label):
    Xtr_raw, ytr = build(train_m, train_h)
    Xte_raw, yte = build(test_m, test_h)
    # standardise on TRAIN statistics only
    Xtr, mean, sd = standardise(Xtr_raw)
    Xte = [[(r[j] - mean[j]) / sd[j] for j in range(len(mean))] for r in Xte_raw]
    w, b = fit(Xtr, ytr)
    sc = [sum(w[j] * xi[j] for j in range(len(w))) + b for xi in Xte]
    a = auc(sc, yte)
    acc = sum(1 for s, t in zip(sc, yte) if (s > 0) == (t == 1)) / len(yte)
    print("  %-34s AUC %.3f   acc %.3f   (train %d, test %d)" % (
        label, a, acc, len(ytr), len(yte)))
    return a

print("\nheld-out generator:")
run(gem, hA, qwn, hB, "train Gemma-4-26B -> test Qwen3.6-35B")
run(qwn, hA, gem, hB, "train Qwen3.6-35B -> test Gemma-4-26B")

print("\nsame-generator baseline (train/test split within one):")
g1, g2 = gem[:len(gem)//2], gem[len(gem)//2:]
run(g1, hA, g2, hB, "train Gemma -> test Gemma")
