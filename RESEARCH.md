# Research notes

Background for `claudeify`. Three questions: how machine prose is detected, what
the markers actually are, and whether Anthropic's watermark can be reproduced
locally. The short answer to the third one is no, and that shapes the design.

---

## 1. How LLM prose gets detected

Three independent families. They fail in different ways, which matters.

### a. Predictability / perplexity

The oldest and the only one with a real theory behind it. Run the candidate text
through a reference language model and measure how surprised it is. Machine text
is low-surprisal by construction — it was sampled from near the mode of a
distribution, so another model of similar training predicts it well. Human text
spikes: odd word choices, abandoned clauses, unexpected specifics.

Two derived signals do most of the work:

- **Mean token surprisal** (log of perplexity). Lower means more machine-like.
- **Burstiness** — variance, both of token surprisal and of sentence length.
  Humans are bursty. Models are metronomic. A four-word sentence next to a
  forty-word one is a human signature.

This is the family `detect.py` implements. This build of llama.cpp will not
return prompt logprobs — `echo`, `prompt_logprobs`, and `n_probs` with
`n_predict: 0` all come back empty — so the scorer recovers them by sampled
teacher-forcing: tokenize, walk ~24 positions, ask for the top-20 next-token
distribution at each, and look up what the text actually did next. Ascending
positions keep llama.cpp's prefix cache warm, so the whole pass is one prompt
ingest rather than 24.

### b. Trained classifiers

A binary classifier fine-tuned on human/AI corpora. This is what most commercial
detectors ship. It works well in-distribution and degrades badly out of it. The
documented failure mode matters: **false-positive rates above 60% on non-native
English writers**, because L2 English shares the low-perplexity, formulaic
surface that the classifier learned to associate with machines.

### c. Watermarking

The only family that can be strong, and the only one that requires the
generator's cooperation. Covered in section 3.

Surface stylometry — the em-dash-and-"delve" layer — is not really what (a) and
(b) key on. But it is what **human readers and cheap detectors** key on, and for
this tool that is the target.

---

## 2. The marker taxonomy

Assembled from the `avoid-ai-writing` audit lists (inverted — that skill removes
these, `claudeify` adds them), corpus work on punctuation frequencies, and the
observed behaviour of the local models.

**Tier 1 — flagged on sight**
`delve`, `leverage`, `harness`, `tapestry`, `testament`, `realm`, `navigate`,
`underscore`, `pivotal`, `crucial`, `robust`, `nuanced`, `holistic`, `seamless`,
`foster`, `myriad`, `plethora`, `intricate`, `multifaceted`, `landscape`,
`comprehensive`, `cornerstone`, `paradigm`, `streamline`, `empower`.

**Template phrasing**
"It's not just X", "pave the way", "in today's …", "when it comes to", "at its
core", "it's worth noting", "plays a crucial role", "a wide range of", "more
than just", "the intersection of", "In conclusion", "Ultimately,".

**Assistant register**
"I'd say", "happy to", "if that's useful", "let me know", "you might want to",
"your best bet", "I hope this helps", "Great question".

**Constructions**
- *Antithesis*: "It's not just X — it's Y." "Less about X than Y." "Not because
  X, but because Y." The single most recognisable rhythm in current assistant prose.
- *Tricolon*: "A, B, and C." The rule of three, applied compulsively.
- *Signposting*: "Two things are going on here." "The short answer is." "That said,".
- *Hedge stacking*: "could potentially", "generally tends to", "may in some cases".

**Punctuation and shape**
- Em dashes far above human base rate.
- Bulleted lists with **bold lead-ins**.
- Paragraphs of near-identical length.
- Sentences clustered tightly around 15–25 words — the low-burstiness signature.

**The caveat that comes with all of it**: every one of these appears in human
writing too, especially under deadline, in technical genres, and in L2 English.
These are signals, not proof, in both directions.

### Vendor-specific tics, and why they get their own profile

Everything above is *generic* machine prose. One vendor's voice is a different
target, and the community catalogues it far better than any paper does —
r/ClaudeAI, r/ClaudeCode and HN have running threads on it.

The named inventory: **"load-bearing"** (by a wide margin the most-cited),
"worth stating plainly", "carry the argument", "full stop", "the real question
is", "the honest answer/take", "the key insight", "push back", "genuinely",
"synthesize", "and the trap is", "that's the tell", "You're absolutely right
to…", "by design". Structurally: em dashes, inline and nested parentheticals,
"But" opening a counterpoint, and concede-then-disagree.

**The finding that changed the design:** the most-cited structural tic is the
*punchy fragment* — "Not a detail. A design decision." "Full stop." That is
**high** burstiness. But high burstiness is exactly what the generic detector
reads as *human*, because uniform sentence length is the generic machine tell.

So the two goals genuinely conflict on one axis:

| | generic "reads as AI" | Claude's voice |
|---|---|---|
| Sentence length | uniform, 15–25 words | violently varied, fragments |
| Burstiness | low | high |
| Hedging | heavy, unresolved | hedge then commit |
| Vocabulary | delve, leverage, holistic | load-bearing, the tell, full stop |

Hence two scoring profiles rather than one score. `generic` keeps burstiness
scored backwards; `claude` flips it, adds the Claudism lexicon at the heaviest
weight, and rewards fragments and parentheticals.

**Volatility warning:** the generic markers are properties of instruction-tuning
in general and drift slowly. The vendor list tracks one company's current
post-training and will rot — Anthropic has already shipped banned-word lists in
response to exactly this feedback. Treat `CLAUDISMS` in `detect.py` as a
maintained list, not a constant.

---

## 3. Anthropic's text watermark

### What is publicly known

- Anthropic announced invisible watermarking of Claude's text and image output,
  with reporting placing embedding in Claude models from **2 August 2026**. The
  driver is provenance obligations under the EU AI Act Code of Practice.
- The mechanism, as described in the independent write-ups, is **statistical
  token biasing** in the lineage of Kirchenbauer et al. (ICML 2023): generation
  is nudged to "subtly favor certain possible tokens according to a secret rule."
  No single choice is unusual; over enough text the accumulated preference
  becomes a signal a hypothesis test can pick up.
- **It adds nothing to the string.** There are no zero-width characters, no
  Unicode tags, no hidden markup. The watermark is a property of *which words
  were chosen*, not of extra bytes carried alongside them.
- Anthropic has not published full technical detail; documentation on the
  detection mechanism was described as forthcoming.
- Anthropic's own framing is deliberately weak in both directions: a hit means
  the content **may have been processed by Claude**, and a miss **does not**
  prove no AI was involved.

### The cottage industry that has this wrong

A large number of "remove the Claude watermark" sites offer to strip invisible
Unicode from your text. Against a sampling watermark this does nothing — there
is nothing there to strip. Some of those same pages now say so themselves. What
*does* degrade a statistical watermark is heavy paraphrase, translation, and
short length, which is exactly what the SynthID-Text documentation says about
its own robustness limits.

### Why claudeify cannot reproduce it — and should not try

A watermark of this kind lives in **the sampler of the model that generated the
text**, keyed by a secret the generator holds. Two consequences:

1. Your local Qwen sampler cannot emit Anthropic's watermark. You do not have
   the key, and no amount of stylistic imitation produces one. Style and
   watermark are unrelated channels.
2. If it were possible, it would be forging a provenance signal — attaching
   "Anthropic generated this" to text Anthropic did not generate. That is the
   one attack watermarking exists to prevent, and it would poison the signal for
   everyone relying on it.

So the tool imitates the **voice**, and is honest about the **provenance**.

### What a real machine-readable "this is AI" signal looks like locally

If you want more than style, these are the honest options, in descending order
of strength:

| Option | Strength | Cost |
|---|---|---|
| **SynthID-Text with your own key** — open, in `transformers` ≥ 4.46 (`SynthIDTextWatermarkLogitsProcessor` + detector) | Real statistical watermark, survives light editing | Needs a HF `generate()` loop, not llama.cpp; detection needs a trained Bayesian detector (~10k samples) |
| **Invisible Unicode tag** (`convert.py --invisible`) | Self-declared, machine-readable, trivially strippable | Free; already implemented |
| **Visible disclosure line** (`--disclosure`) | Unambiguous, survives everything | Free; visible |

The tag marker spells `AI-GENERATED` in the Unicode TAG block (U+E0000). It
claims only what is true — this text came out of this tool — and imitates no
vendor's scheme.

---

## 4b. RAID, and what fitting to it does and does not buy

**RAID** — "Robust AI Detection", Dugan et al., ACL 2024 — is the standard shared
benchmark for machine-generated-text detection, and the answer to "where would
objective weights come from". It ships ~2.27M labelled training rows: **8 domains**
(abstracts, books, news, poetry, recipes, reddit, reviews, wiki), **11 generators**
(gpt2, gpt3, chatgpt, gpt4, mistral, mistral-chat, mpt, mpt-chat, cohere,
cohere-chat, llama-chat), 4 decoding strategies, and **11 adversarial attacks**
(paraphrase, homoglyph, whitespace insertion, synonym swap …). Its whole point is
that detectors which look excellent in-distribution collapse on unseen generators
and under adversarial edits, so results are conventionally reported as recall at
a 5% false-positive rate rather than raw accuracy.

`fit_weights.py` fits a logistic regression on 2,400 balanced rows of it —
unattacked only, quota'd per domain and per generator — using nothing but the
features `detect.py` already computes. Pure stdlib, so nothing gains a dependency.

**Result: 5-fold CV AUC 0.746, accuracy 0.715 (sd 0.011).** Surface style alone
is a weak-but-real detector, which is the honest expected outcome; production
detectors reach AUC 0.9+ by adding perplexity and a trained transformer.

What the fit settles:

- **Burstiness direction confirmed.** `sentence_len_cv` is the second-strongest
  feature and negative — higher variance means human. The hand-picked ramp had
  the sign right.
- **Bullets are the single strongest surface feature** (24.1% of total weight),
  far above the 6 I had guessed.
- **Em dashes are nearly worthless here** (0.6%) against the 11 I assigned. The
  most culturally notorious tell is, on this data, almost no evidence at all.

### …and then it failed to transfer

Scored on this tool's own output, the RAID fit runs **backwards**: a raw human
note scored 47.1 and its restyled version 7.3. The contributions say why —
`hedge_per_1k` at **−1.155** and `claudism_per_1k` at **−1.059**, both pushing
*toward human*.

RAID's generators are 2022–23. They do not hedge like a 2026 RLHF'd assistant, so
in that corpus "generally", "tends to", "roughly" and "worth noting" genuinely are
human markers — they show up in reddit and reviews, not in GPT-2. The fit learned
that faithfully, and it is exactly wrong for the text this tool produces.

This is ordinary distribution shift, and it is the thing to hold onto: **an
objective fit is objective about its own corpus.** RAID makes the weights
defensible for detecting 2023-era generators across 8 public domains. It does not
make them right for scoring 2026 assistant-register prose, and swapping in a
number with an AUC attached would have quietly made the tool worse while looking
more rigorous.

### Refitting on 2026 generators

The vintage hypothesis was testable, so I tested it: generate a machine corpus
with a current model (Gemma-4-26B-A4B, 200 texts across RAID's 8 domains, seeded
from RAID's own human texts) and refit against the same human rows.

**AUC 0.746 → 0.938. Accuracy 0.715 → 0.868.** And it transfers: the note that
went 47.1 → 7.3 backwards under the RAID fit now goes **14.0 → 100.0**.

The coefficient diff says exactly what changed:

| feature | RAID (2022-23) | modern (2026) | |
|---|---|---|---|
| `bold_runs` | −0.004 | **+1.741** | strongest modern feature; old models emitted no markdown |
| `emdash_per_1k` | +0.018 | **+0.622** | worthless on RAID, 3rd strongest now |
| `paragraph_len_cv` | −0.010 | +0.410 | appears only in modern |
| `hedge_per_1k` | −0.076 | +0.002 | flips sign, but lands at ~zero: uninformative, not a tell |
| `fragment_rate` | +0.153 | −0.072 | sign flip — fragments now read as human |
| `sentence_len_cv` | −0.546 | −0.509 | **stable across both corpora** |
| `bullet_lines` | +0.732 | +0.889 | strong in both |

So the em-dash reputation is earned — just not by the models RAID was built on.
Markdown bold and em dashes are post-2024 RLHF artifacts, and a benchmark from
2024 cannot see them. Two findings survive both corpora and are the ones to
trust: **low sentence-length variance and bullet density**.

### Is it just learning one model?

A fit trained on the same model the tool generates with could be learning that
model's quirks rather than machine-ness — the exact failure RAID was built to
expose. So: hold a generator out entirely.

| | AUC | accuracy |
|---|---|---|
| train Gemma-4-26B → test Qwen3.6-35B | **0.921** | 0.857 |
| train Qwen3.6-35B → test Gemma-4-26B | **0.908** | 0.840 |
| same-generator baseline (Gemma → Gemma) | 0.878 | 0.805 |

Cross-generator performance **matches** same-generator, so the signal is modern
machine style in general, not one family's fingerprint. (The same-generator
baseline scores slightly lower only because it trains on half the rows.)

The shipped model is fitted on both generators combined — 680 rows, 8 domains —
at **AUC 0.949, accuracy 0.884**, weights: bold runs 32%, em dashes 13%, bullet
lines 12%, sentence-length variance 10% (negative), paragraph variance 8%,
tier-1 vocabulary 8%.

Sanity on the shipped model end to end:

```
raw note      detectability  17.5    restyled   99.9
human terse   detectability  10.7    human long 20.4
```

Note also that `claudism_per_1k` stays *negative* in both fits. The Claudism
lexicon is not a machine-vs-human discriminator at all — it identifies one
vendor's voice, which is precisely why it lives in the `claude` profile and not
in the generic detector.

## 5. Prior art

Worth knowing what exists, because the market runs almost entirely the other way.

- **Humanizers** — Undetectable.ai, Humbot, StealthGPT, WriteHuman, and a long
  tail of GitHub "AI text humanizer" repos. These strip the markers to defeat
  detectors. Same feature set, opposite sign, and a much larger market.
- **`conorbronsdon/avoid-ai-writing`** — the closest thing to a direct inverse:
  a skill that audits prose against a P0/P1/P2 tell taxonomy and rewrites them
  out. Its lists are the single best source for what to put *in*.
- **Detectors** — GPTZero, Originality.ai, Pangram, Turnitin. The scoring half.
- **Academic** — DetectGPT (curvature), Binoculars (cross-perplexity), and the
  RAID benchmark. This is where the perplexity family is done properly.
- **Style-transfer research** — mostly framed as *authorship obfuscation*, i.e.
  removing a signal rather than adding one.

Searches for a tool doing the forward direction — deliberately making text read
as machine-written — returned nothing. There is no commercial demand for it,
which is presumably why. The nearest neighbours are joke "ChatGPT-ify" prompts
and the parody accounts.

## Sources

- Kirchenbauer et al., *A Watermark for Large Language Models*, ICML 2023 — the
  token-biasing scheme Claude's watermark is reported to follow.
- Dathathri et al., *Scalable watermarking for identifying large language model
  outputs*, Nature 634 (Oct 2024) — SynthID-Text, tournament sampling.
- Hugging Face, *Introducing SynthID Text* — the open implementation and its
  stated limits (weak on factually constrained output, degrades under rewriting).
- Columbia Statistics (T. Zheng), *Claude Is Getting an LLM Watermark* (Aug 2026)
  — mechanism, and the caveat that Anthropic has not confirmed the specific scheme.
- PCMag, *Anthropic's Claude Will Now Add Invisible Watermarks* (Aug 2026).
- `conorbronsdon/avoid-ai-writing` — the P0/P1/P2 audit taxonomy, inverted here,
  including the >60% L2 false-positive figure.
