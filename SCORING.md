# How the score is computed

Everything here is in `detect.py`; this page says what it does and, just as
important, what has and hasn't been checked. The score is **a rehearsal, not a
verdict from a real detector.**

## The number you see

```
headline = 0.62 * style_score + 0.38 * perplexity_score      (perplexity pass on)
headline = style_score                                        (perplexity pass off or unavailable)
```

| headline | verdict shown |
|---|---|
| >= 78 | very likely flagged as AI-generated |
| 62 - 78 | likely flagged as AI-generated |
| 45 - 62 | borderline |
| < 45 | likely reads as human |

`style_score` comes from surface features of the text (below). `perplexity_score`
comes from a small local model (below). The verdict words are thresholds on a
number, not a classifier's output.

## Two kinds of style score

The same 23 raw measurements are computed for every text (`raw` in the JSON and
the "Measurements" table in the UI). What differs is how they are weighted.

**Generic voice: a fitted logistic regression.** 18 of the measurements go through
a model fitted on 680 balanced rows: human text from RAID (Dugan et al., ACL 2024)
against text from two 2026 generators (Gemma-4-26B-A4B, Qwen3.6-35B-A3B). Each
feature is standardised, multiplied by its coefficient, summed with an intercept,
and passed through a sigmoid. Largest coefficients (`fitted-weights.json`):

| feature | coef | reads as |
|---|---|---|
| bold runs | +1.85 | machine |
| em dashes per 1k words | +0.76 | machine |
| bullet lines | +0.71 | machine |
| sentence-length variation | -0.56 | uneven sentences = human |
| paragraph-length variation | +0.48 | machine |
| tier-1 vocabulary per 1k | +0.46 | machine |
| tricolons per 1k | +0.28 | machine |
| antithesis per 1k | +0.17 | machine |

Hedging, second person, semicolons and the Claude-tic lexicon all sit near zero.
5-fold CV AUC 0.949, accuracy 0.884; holding a generator out entirely gives AUC
0.91-0.92. The output is a probability, so it saturates: median 98.3 on generated
text.

**Claude voice (the default): hand-picked weights.** There is no labelled corpus of
one vendor's prose, so nothing is fitted. Eleven features, each a clamped linear
ramp between a low and a high per-1000-word rate, combined by weight:

| feature | weight | | feature | weight |
|---|---|---|---|---|
| contractions | 20 | | fragments (<= 5 words) | 8 |
| first person | 15 | | no passive openers | 8 |
| Claude-tic phrases | 12 | | parentheticals | 5 |
| antithesis | 10 | | "But" openers | 5 |
| em dashes | 9 | | tricolons / second person | 4 each |

The ramps came from comparing genuine Claude prose with this tool's own output
(contractions 35 vs 10.6 per 1k, first person 29 vs 3.6). They were not validated
against any external detector.

**Style density.** The generic voice also has a hand-weighted ten-or-eleven-feature version
(tier-1 vocabulary 14, low burstiness 14, template phrasing 13, ...). The retry
loop targets this number, not the fitted probability, because the probability
saturates and a target the text already passes would never trigger a retry. The
Claude voice uses its hand-weighted score for both jobs.

## The perplexity pass

A small local llama.cpp model (Qwen 3.5 2B in the author's setup) reads the text. At 24 evenly spaced
positions (28 for `/api/score`) it is shown everything before that point and asked for its top 20
guesses for the next token. Three numbers come out:

- mean log-probability of the token that was actually there (higher = more predictable),
- how often the actual token was the model's first guess (top-1 agreement),
- the spread of those log-probabilities.

```
perplexity_score = 100 * (0.45 * f_logprob + 0.35 * f_top1 + 0.20 * f_spread)
```

where each `f` is a clamped ramp (log-prob from -3.6 to -0.9, top-1 from 0.20 to
0.62, spread from 3.0 down to 1.1). Predictable, even text scores high. A token
outside the top 20 is charged the 20th-best log-probability minus 1.5 rather than
dropped. Texts under 12 tokens skip the pass. The pass does not steer retries; it
only moves the headline.

## What has been checked, and what hasn't

- **Checked:** the fitted model separates RAID human text from modern machine text
  (AUC 0.949, and across generators). `RESEARCH.md` section 4b has the account,
  including why a fit on 2022-23 generators failed to transfer.
- **Checked, negative:** Wikipedia's "Signs of AI writing" markers add +0.0004 AUC and
  were not adopted. Hedging, long assumed to be a tell, carries almost no weight.
- **Checked against a real detector, and it missed:** on ten Pangram checks the
  headline score agreed with Pangram on the tool's own generic output but missed
  plain LLM rewrites and Claude-voice output (Pangram read those as human or
  AI-assisted). See `RESEARCH.md`, "An external check: Pangram". Short texts also
  limit Pangram's own confidence.
- **Not checked:** the perplexity pass on its own against any detector, the
  hand-picked Claude-voice weights, and any text over about 250 words.

So read the number as "how much this text looks like the machine prose the model
was fitted on" (generic voice) or "how many of the surface habits I measured in
Claude's writing it shows" (Claude voice). Neither is a prediction of what a given
detector will say.

## Looking inside a score

- UI: the bars show each feature's contribution; "Measurements" lists the raw values.
- API: `POST /api/score` with `{"text": "...", "use_judge": false, "profile": "claude"}`
  returns `score`, `verdict`, `stylometric.raw`, `stylometric.contributions` and
  `predictability`.
- CLI: `python3 convert.py --json "text"` prints the same report for the output.
- Rebuild the fit with `python3 fit_weights.py corpus/modern-combined.jsonl`;
  `python3 crossgen_eval.py` reruns the held-out-generator check (the corpus is
  rebuilt locally, see `ATTRIBUTION.md`).
