"""
detect.py -- offline "does this read as AI-generated?" scorer.

Two independent signal families, both fully local:

  1. STYLOMETRIC  (pure Python, no model)
     Surface markers that public AI-detectors and human readers actually key
     on: tier-1 vocabulary, discourse scaffolding, em-dash rate, antithesis
     ("not X, but Y"), tricolons, hedge density, and burstiness (variance of
     sentence length -- humans are bursty, models are metronomic).

  2. PREDICTABILITY  (needs a local llama.cpp server)
     The perplexity family that DetectGPT / GPTZero-style tools are built on.
     This build of llama.cpp will not return prompt logprobs, so we recover
     them by sampled teacher-forcing: tokenize, walk a set of positions, and
     at each one ask the model for its top-k next-token distribution and look
     up what the text actually did next.  Prompt caching makes the ascending
     prefixes cheap.

Higher score = more likely to be flagged as machine-written.  That is the
goal here, not the thing to avoid.
"""

import json
import math
import os
import re
import urllib.request

# --------------------------------------------------------------------------
# lexicons
# --------------------------------------------------------------------------

# Tier 1: the words public detectors and editors flag on sight.
TIER1 = [
    "delve", "leverage", "harness", "tapestry", "testament", "realm",
    "navigate", "navigating", "underscore", "underscores", "pivotal",
    "crucial", "robust", "nuanced", "holistic", "seamless", "seamlessly",
    "foster", "fostering", "myriad", "plethora", "intricate", "multifaceted",
    "landscape", "embark", "unlock", "elevate", "resonate", "meticulous",
    "comprehensive", "cornerstone", "beacon", "vibrant", "ever-evolving",
    "game-changer", "paradigm", "synergy", "streamline", "empower",
    "transformative", "profound", "compelling", "invaluable", "vital",
]

# Multi-word template phrases.
TEMPLATES = [
    "it's not just", "it is not just", "pave the way", "in today's",
    "in the world of", "when it comes to", "at its core", "the key takeaway",
    "it's worth noting", "it is worth noting", "plays a crucial role",
    "a wide range of", "more than just", "the intersection of",
    "let's dive in", "let's explore", "in conclusion", "ultimately,",
    "that said", "here's the thing", "the short answer",
    "there's a real question", "a few things are going on",
    "the honest answer", "to be clear", "in practice", "the tricky part",
    "worth flagging", "the useful framing",
]

# Assistant-register tells: hedged, second-person, service-shaped.
ASSISTANT = [
    "i'd say", "i would say", "i think it's worth", "happy to",
    "if you want", "if that's useful", "let me know", "you might",
    "you'll want", "your best bet", "the thing to watch",
    "i hope this helps", "great question", "absolutely",
]

# Claude-specific tics, as catalogued by r/ClaudeAI, r/ClaudeCode and HN --
# distinct from generic assistant register. Volatile: these track one vendor's
# current post-training and will drift with each model revision.
CLAUDISMS = [
    "load-bearing", "load bearing", "worth stating plainly", "carry the argument",
    "full stop", "the real question", "the honest take", "the honest answer",
    "to be honest", "honestly", "genuinely", "genuine", "synthesize",
    "key insight", "push back", "pushback", "the tell", "and the trap",
    "you're absolutely right", "you are absolutely right", "let me be direct",
    "here's the thing", "to be clear", "the short answer", "the thing is",
    "caveat", "that said", "worth noting", "the useful framing",
    "matters more than", "not a detail", "by design",
]

HEDGES = [
    "may", "might", "could", "generally", "typically", "often", "somewhat",
    "relatively", "fairly", "tends to", "tend to", "likely", "perhaps",
    "arguably", "broadly", "largely", "roughly", "usually", "in most cases",
    "to some extent", "more or less", "reasonably",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")

# "not X, but Y" / "isn't just X -- it's Y" / "less about X than Y"
_ANTITHESIS = re.compile(
    r"\b(?:not\s+(?:just|only|so\s+much)\b[^.!?]{0,90}?\b(?:but|it's|its|so\s+much\s+as)\b"
    r"|isn'?t\s+(?:just|only)\b[^.!?]{0,90}?\b(?:it'?s|but)\b"
    r"|less\s+about\b[^.!?]{0,90}?\bthan\b"
    r"|not\s+because\b[^.!?]{0,90}?\bbut\s+because\b)",
    re.I,
)

# "A, B, and C" -- the rule of three.
_TRICOLON = re.compile(
    r"\b\w[\w'-]*(?:\s+\w[\w'-]*){0,3},\s+\w[\w'-]*(?:\s+\w[\w'-]*){0,3},\s+"
    r"(?:and|or)\s+\w[\w'-]*",
    re.I,
)


_PUNCT_MAP = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x2032: "'", 0x2033: '"',
}


def normalise(text):
    """Fold typographic quotes to ASCII so the lexicons actually match."""
    return (text or "").translate(_PUNCT_MAP)


def _words(text):
    return _WORD.findall(text)


def _sentences(text):
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    return [s for s in _SENT_SPLIT.split(flat) if s.strip()]


def _count_phrases(low, phrases):
    return sum(low.count(p) for p in phrases)


def _ramp(value, lo, hi):
    """0.0 at or below lo, 1.0 at or above hi."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


# --------------------------------------------------------------------------
# family 1: stylometry
# --------------------------------------------------------------------------

def stylometrics(text, profile="generic"):
    # Em dashes and sentence structure are read from the original; lexicon
    # matching is done on the apostrophe-folded copy.
    text = normalise(text)
    low = text.lower()
    words = _words(text)
    n = max(len(words), 1)
    per_k = lambda c: 1000.0 * c / n           # noqa: E731
    sents = _sentences(text)

    lens = [len(_words(s)) for s in sents if _words(s)]
    if len(lens) >= 3:
        mean = sum(lens) / len(lens)
        var = sum((x - mean) ** 2 for x in lens) / (len(lens) - 1)
        cv = math.sqrt(var) / mean if mean else 0.0
    else:
        mean, cv = (sum(lens) / len(lens) if lens else 0.0), 0.55

    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paras) >= 3:
        plens = [len(_words(p)) for p in paras]
        pmean = sum(plens) / len(plens)
        pvar = sum((x - pmean) ** 2 for x in plens) / (len(plens) - 1)
        pcv = math.sqrt(pvar) / pmean if pmean else 0.0
    else:
        pcv = None

    # Fragments: Claude's most-cited and least-discussed tic ("Not a detail.
    # A design decision."). Generic assistant prose does not do this.
    frag = sum(1 for L in lens if L <= 5)
    frag_rate = frag / len(lens) if lens else 0.0

    # Measured against genuine Claude prose: contractions 35/1k vs 10.6 in the
    # tool's own output, first person 29.1 vs 3.6, and "It appears that" 0.0 vs
    # 17.1. Contractions and first person are the strongest positive markers;
    # passive nominal openers are a strong NEGATIVE one.
    n_contr = len(re.findall(r"\b\w+['\u2019](?:s|t|re|ve|ll|d|m)\b", text, re.I))
    n_first = len(re.findall(r"\bI\b|\bI['\u2019]", text))
    n_passive = len(re.findall(
        r"(?:^|(?<=[.!?]) )(?:It (?:appears|is|was|seems)|This (?:behaviou?r|issue|process|pattern) is)\b",
        text))

    raw = {
        "claudism_per_1k":  per_k(_count_phrases(low, CLAUDISMS)),
        "contraction_per_1k": per_k(n_contr),
        "first_person_per_1k": per_k(n_first),
        "passive_opener_per_1k": per_k(n_passive),
        "fragment_rate":    round(frag_rate, 3),
        "paren_per_1k":     per_k(len(re.findall(r"\([^)]{3,}\)", text))),
        "but_opener_per_1k": per_k(len(re.findall(r"(?:^|(?<=[.!?]) )But\b", text))),
        "tier1_per_1k":      per_k(sum(len(re.findall(r"\b%s\b" % re.escape(w), low)) for w in TIER1)),
        "template_per_1k":   per_k(_count_phrases(low, TEMPLATES)),
        "assistant_per_1k":  per_k(_count_phrases(low, ASSISTANT)),
        "hedge_per_1k":      per_k(sum(len(re.findall(r"\b%s\b" % re.escape(w), low)) for w in HEDGES)),
        "emdash_per_1k":     per_k(text.count("—") + len(re.findall(r"\s--\s", text))),
        "semicolon_per_1k":  per_k(text.count(";")),
        "antithesis_per_1k": per_k(len(_ANTITHESIS.findall(text))),
        "tricolon_per_1k":   per_k(len(_TRICOLON.findall(text))),
        "second_person_per_1k": per_k(len(re.findall(r"\b(?:you|your|you're|you'll)\b", low))),
        "bullet_lines":      len(re.findall(r"^\s*(?:[-*•]|\d+\.)\s+", text, re.M)),
        "bold_runs":         len(re.findall(r"\*\*[^*]+\*\*", text)),
        "sentence_len_mean": round(mean, 1),
        "sentence_len_cv":   round(cv, 3),
        "paragraph_len_cv":  round(pcv, 3) if pcv is not None else None,
        "n_words":           len(words),
        "n_sentences":       len(sents),
    }

    # Each contribution is a saturating ramp on a per-1000-word rate, so the
    # score does not drift with document length.
    #
    # Two profiles, because the targets genuinely differ. "generic" is what
    # public detectors and readers flag as machine-written: uniform, hedged,
    # metronomic. "claude" is one vendor's voice, which is NOT metronomic --
    # its most-cited tic is the punchy fragment, so burstiness flips sign.
    if profile == "claude":
        contrib = {
            "contractions":        (_ramp(raw["contraction_per_1k"], 5, 35), 20),
            "first_person":        (_ramp(raw["first_person_per_1k"], 2, 25), 15),
            "claudisms":           (_ramp(raw["claudism_per_1k"], 0, 14), 12),
            "antithesis":          (_ramp(raw["antithesis_per_1k"], 0, 5), 10),
            "em_dash_rate":        (_ramp(raw["emdash_per_1k"], 0, 10), 9),
            "fragments":           (_ramp(raw["fragment_rate"], 0.0, 0.22), 8),
            "parentheticals":      (_ramp(raw["paren_per_1k"], 0, 7), 5),
            "but_openers":         (_ramp(raw["but_opener_per_1k"], 0, 6), 5),
            "tricolon":            (_ramp(raw["tricolon_per_1k"], 0, 6), 4),
            "second_person":       (_ramp(raw["second_person_per_1k"], 0, 25), 4),
            # Negative marker: generic-assistant passive openers, which genuine
            # Claude prose uses at a measured rate of zero.
            "no_passive_openers":  (1.0 - _ramp(raw["passive_opener_per_1k"], 0, 12), 8),
        }
    else:
        contrib = {
            "tier1_vocabulary":    (_ramp(raw["tier1_per_1k"], 0, 12), 14),
            "template_phrasing":   (_ramp(raw["template_per_1k"], 0, 10), 13),
            "assistant_register":  (_ramp(raw["assistant_per_1k"], 0, 6), 9),
            "hedging":             (_ramp(raw["hedge_per_1k"], 4, 28), 11),
            "em_dash_rate":        (_ramp(raw["emdash_per_1k"], 0, 9), 11),
            "antithesis":          (_ramp(raw["antithesis_per_1k"], 0, 5), 10),
            "tricolon":            (_ramp(raw["tricolon_per_1k"], 0, 6), 7),
            "second_person":       (_ramp(raw["second_person_per_1k"], 0, 25), 5),
            # Low variance is the AI signal, so this ramp runs backwards.
            "low_burstiness":      (1.0 - _ramp(raw["sentence_len_cv"], 0.32, 0.78), 14),
            "structural_scaffold": (_ramp(raw["bullet_lines"] + raw["bold_runs"], 0, 8), 6),
        }
        if raw["paragraph_len_cv"] is not None:
            contrib["uniform_paragraphs"] = (1.0 - _ramp(raw["paragraph_len_cv"], 0.20, 0.70), 5)

    total_w = sum(w for _, w in contrib.values())
    score = 100.0 * sum(f * w for f, w in contrib.values()) / total_w
    return {
        "score": round(score, 1),
        "profile": profile,
        "raw": raw,
        "contributions": {k: {"strength": round(f, 3), "weight": w,
                              "points": round(100.0 * f * w / total_w, 1)}
                          for k, (f, w) in sorted(
                              contrib.items(), key=lambda kv: -kv[1][0] * kv[1][1])},
    }


# --------------------------------------------------------------------------
# fitted weights (logistic regression on RAID; see fit_weights.py)
# --------------------------------------------------------------------------

_FITTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fitted-weights.json")
_FITTED = None


def load_fitted(path=None):
    """Load fitted coefficients if present. Absent file => heuristic scoring."""
    global _FITTED
    if _FITTED is None:
        try:
            with open(path or _FITTED_PATH) as f:
                _FITTED = json.load(f)
        except Exception:
            _FITTED = False
    return _FITTED or None


def fitted_score(raw, model=None):
    """
    Score from the fitted logistic model instead of hand-picked weights.

    Returns None when no fit is available. The contributions are signed: a
    negative one means that feature is pushing the text toward "human".
    """
    m = model or load_fitted()
    if not m:
        return None
    z, contrib = [], {}
    logit = m["intercept"]
    for i, name in enumerate(m["features"]):
        x = raw.get(name)
        if x is None:
            x = 0.45
        zi = (float(x) - m["mean"][i]) / (m["sd"][i] or 1.0)
        c = m["coef"][i] * zi
        logit += c
        z.append(zi)
        contrib[name] = c
    p = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
    tot = sum(abs(v) for v in contrib.values()) or 1.0
    return {
        "score": round(100.0 * p, 1),
        "probability": round(p, 4),
        "logit": round(logit, 3),
        "cv_auc": m.get("cv_auc"),
        "cv_acc": m.get("cv_acc"),
        "n_train": m.get("n"),
        "contributions": {
            k: {"strength": round(abs(v) / tot, 3),
                "weight": round(100 * abs(m["coef"][m["features"].index(k)])
                                / (sum(abs(c) for c in m["coef"]) or 1.0), 1),
                "points": round(v, 3),
                "pushes": "machine" if v > 0 else "human"}
            for k, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1]))},
    }


# --------------------------------------------------------------------------
# family 2: predictability (perplexity by sampled teacher-forcing)
# --------------------------------------------------------------------------

def _post(url, payload, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def predictability(text, base="http://127.0.0.1:6770", model="qwen3.5-2b",
                   samples=28, topk=20, timeout=120):
    """
    Estimate mean next-token surprisal over `samples` positions.

    Returns None if the server is unreachable or the text is too short --
    predictability is an optional signal, never a hard dependency.
    """
    try:
        toks = _post(base + "/tokenize",
                     {"content": text, "with_pieces": True}, timeout)["tokens"]
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200]}

    ids = [t["id"] for t in toks]
    if len(ids) < 12:
        return {"available": False, "reason": "text too short (<12 tokens)"}

    # Skip the first few tokens: nothing to condition on, so their surprisal
    # is noise. Ascending positions keep the llama.cpp prefix cache warm.
    lo, hi = 4, len(ids) - 1
    n = min(samples, hi - lo)
    step = (hi - lo) / n
    positions = sorted({int(lo + i * step) for i in range(n)})

    logprobs, ranks, floor_hits = [], [], 0
    for pos in positions:
        try:
            d = _post(base + "/completion", {
                "prompt": ids[:pos], "n_predict": 1, "n_probs": topk,
                "temperature": 0, "cache_prompt": True,
            }, timeout)
            top = d["completion_probabilities"][0]["top_logprobs"]
        except Exception:
            continue

        actual = ids[pos]
        hit = next(((i, t["logprob"]) for i, t in enumerate(top)
                    if t["id"] == actual), None)
        if hit is None:
            # Censored: the true token fell outside the top-k. Charge it the
            # k-th logprob minus a margin rather than dropping the sample.
            logprobs.append(top[-1]["logprob"] - 1.5)
            ranks.append(topk)
            floor_hits += 1
        else:
            logprobs.append(hit[1])
            ranks.append(hit[0])

    if len(logprobs) < 5:
        return {"available": False, "reason": "too few usable samples"}

    mean_lp = sum(logprobs) / len(logprobs)
    var_lp = sum((x - mean_lp) ** 2 for x in logprobs) / max(len(logprobs) - 1, 1)
    top1 = sum(1 for r in ranks if r == 0) / len(ranks)
    top5 = sum(1 for r in ranks if r < 5) / len(ranks)

    # Predictable text = mean logprob near 0, high top-1 agreement, and low
    # spread. All three point the same way.
    f_lp = 1.0 - _ramp(-mean_lp, 0.9, 3.6)
    f_t1 = _ramp(top1, 0.20, 0.62)
    f_var = 1.0 - _ramp(math.sqrt(var_lp), 1.1, 3.0)
    score = 100.0 * (0.45 * f_lp + 0.35 * f_t1 + 0.20 * f_var)

    return {
        "available": True,
        "score": round(score, 1),
        "mean_logprob": round(mean_lp, 3),
        "perplexity_est": round(math.exp(-mean_lp), 2),
        "top1_agreement": round(top1, 3),
        "top5_agreement": round(top5, 3),
        "logprob_sd": round(math.sqrt(var_lp), 3),
        "samples_used": len(logprobs),
        "outside_topk": floor_hits,
        "judge_model": model,
    }


# --------------------------------------------------------------------------
# combined
# --------------------------------------------------------------------------

def score_text(text, use_model=True, base="http://127.0.0.1:6770",
               model="qwen3.5-2b", samples=28, profile="generic",
               weights="fitted"):
    sty = stylometrics(text, profile)

    # The generic profile prefers the fitted model when one is on disk; the
    # claude profile has no fit (no labelled corpus of one vendor's prose), so
    # it always uses the hand-picked weights.
    # Two numbers, because the score does two jobs that pull apart:
    #
    #   detectability  -- "would this be flagged?"  The fitted logistic model
    #       answers this with an AUC behind it, but it is a PROBABILITY and it
    #       saturates: median 98.3 over generated text, 42% at >=99.
    #   style_density  -- "how heavily marked is this?"  The graded heuristic.
    #       Useless as a detector claim, but it is what the retry loop needs,
    #       because a target the text already passes produces no retries.
    #
    # Using the fitted probability to drive retries would silently disable the
    # loop, so the loop keeps the heuristic and the headline keeps the fit.
    fit = fitted_score(sty["raw"]) if (weights == "fitted" and profile == "generic") else None
    sty["style_density"] = sty["score"]
    if fit:
        sty = dict(sty, score=fit["score"], detectability=fit["score"],
                   contributions=fit["contributions"], weights="fitted",
                   cv_auc=fit["cv_auc"], cv_acc=fit["cv_acc"],
                   style_contributions=sty["contributions"])
    else:
        sty = dict(sty, weights="heuristic", detectability=None)
    pred = predictability(text, base, model, samples) if use_model else \
        {"available": False, "reason": "disabled"}

    if pred.get("available"):
        combined = 0.62 * sty["score"] + 0.38 * pred["score"]
    else:
        combined = sty["score"]

    if combined >= 78:
        verdict = "very likely flagged as AI-generated"
    elif combined >= 62:
        verdict = "likely flagged as AI-generated"
    elif combined >= 45:
        verdict = "borderline"
    else:
        verdict = "likely reads as human"

    return {
        "score": round(combined, 1),
        "verdict": verdict,
        "stylometric": sty,
        "predictability": pred,
    }


def weakest_signals(result, k=4):
    """The lowest-scoring stylometric levers, for the retry prompt."""
    c = result["stylometric"]["contributions"]
    return [name for name, _ in sorted(c.items(), key=lambda kv: kv[1]["strength"])[:k]]
