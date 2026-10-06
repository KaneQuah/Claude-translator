# claudeify

**Paste text in, get it back in Claude's voice, using a model that runs on your own machine.**
Your meaning, an assistant's register. No API keys, no cloud calls, no pip installs.

![claudeify web UI showing a converted note and its score](docs/screenshot.jpg)

It restyles; it does not rewrite. The prompt's top rule is *never respond to the
input, only restyle it*, so a rant stays a rant, a question stays a question, and
nonsense stays nonsense, just wearing a different voice. Then it scores the result and
retries until it hits a target, naming the markers the last draft was thin on.

> **What this is not.** It is not a tool for passing machine text off as human, and it
> is not a detector. The score is a rehearsal built from stylometry and a small model's
> predictability, and on a real detector (Pangram) it missed. [SCORING.md](SCORING.md)
> says exactly what is and isn't validated.

## Quick start (Ollama)

You need Python 3 and a model server. Ollama is the easy one.

```bash
# 1. Install Ollama (https://ollama.com/download), then pull a model that fits your GPU:
ollama pull gemma4:12b        # 8-12 GB of VRAM
ollama pull gemma4:26b        # 20 GB+ of VRAM (or slower, with CPU offload)

# 2. Get claudeify and start it
git clone https://github.com/KaneQuah/Claude-translator.git
cd Claude-translator
./run.sh                      # then open http://127.0.0.1:7080
```

That is all. claudeify asks Ollama which models you have and offers them in the
**Model** dropdown, with Gemma 4 26B preselected if you have it.

### Pointing it at your model server

claudeify talks to any server with the OpenAI-compatible API: Ollama, llama.cpp's
`llama-server`, LM Studio, vLLM. Set `CLAUDEIFY_ENDPOINT` to its address (comma-separate
several):

```bash
CLAUDEIFY_ENDPOINT=http://192.168.1.50:11434 ./run.sh       # Ollama on another machine
CLAUDEIFY_ENDPOINT=http://127.0.0.1:1234 ./run.sh           # LM Studio
CLAUDEIFY_MODEL=gemma4:12b ./run.sh                         # preselect a model
```

If Ollama runs on a different machine, start it there with `OLLAMA_HOST=0.0.0.0` so it
listens on the network. The default, when you set nothing, is Ollama on
`127.0.0.1:11434` and `llama-server` on `127.0.0.1:8080`.

### How does it know which models you have?

It asks. When the page loads (results are cached for 10 seconds), claudeify calls
`GET /v1/models` on each endpoint and lists whatever comes back, so a model you `ollama pull`
shows up without any config. It never scans your disk. Embedding models are filtered out by
name, and a server too slow to answer (mid-generation) shows as *busy* but stays selectable.
`/api/health` shows exactly what it found. To pin names or roles by hand, copy
`backends.example.json` to `backends.json`.

### Optional: the perplexity pass (needs llama.cpp)

The score can include a second opinion: a small model reads your text and measures how
predictable each next word was ([what that is](SCORING.md#the-perplexity-pass)). That needs
per-token probabilities for text the model did not write, which **Ollama's API cannot
give** (no `logprobs`, no tokenize endpoint), so this one pass is pinned to
[llama.cpp](https://github.com/ggml-org/llama.cpp) whatever model does the restyling:

```bash
llama-server -m Qwen3.5-2B-Q4_K_M.gguf --port 8080      # any small model; found automatically
# or point at one elsewhere: CLAUDEIFY_JUDGE=http://host:port
```

Without it, the checkbox is greyed out with an explanation and the score uses style
markers alone. A small model other than the restyler is best, because a model judging its
own output finds it unusually predictable. (The judge on `:8080` also shows up in the
model list; don't pick it, a 2B model fails the restyle task.)

### Hardware and model recommendations

Measured on the same four inputs for every model; see [Model choice, measured](#model-choice-measured) below.

| Your GPU memory | Run | Why |
|---|---|---|
| **20 GB or more** | `gemma4:26b` (Gemma 4 26B-A4B, MoE) | Best fidelity measured: 0.76 invented claims per 100 words against 3-5 for the others (n=4). Only ~4B parameters active, so it is fast. |
| **16 GB** | Gemma 4 26B-A4B as a ~4-bit GGUF through llama.cpp (`UD-IQ4_XS`, 12.6 GB of weights) | Fits, barely: 16.2 of 17.2 GB used at an 8k context. Ollama's `gemma4:26b` is a 16-19 GB download and won't fit. |
| **8-12 GB** | `gemma4:12b` | Tied the 26B on style at 20 samples each (47.8 vs 48.7, both ±5). Fidelity was not re-measured, so the 26B may still be more faithful. |
| **No GPU / under 8 GB** | works, with a catch | A 12B on 18 CPU threads took ~38 s a pass. Models of 4B and under tended to pad, drift or return the input unchanged. |

Hardware actually tested: Radeon RX 7800 XT (16 GB), Tesla P100 (16 GB), GeForce RTX 4060
(8 GB). Gemma 4 26B ran at about 3.5 s a pass on both 16 GB cards; the 12B at 1.7 s on the
7800 XT. The 20 GB, 8-12 GB and no-GPU rows are inferred from model sizes and the runs above, not
benchmarked on that hardware. Avoid models whose thinking can't be switched off (one 30B model returned empty
output under every setting).

### What I have and haven't tested

Built and tested against llama.cpp's server. Ollama's documented API matches what claudeify
uses (`/v1/models`, `/v1/chat/completions`, `reasoning_effort` to turn thinking off), and a
request that an endpoint rejects is retried without the extra fields, but **I have not run it
against a live Ollama**. If it misbehaves there, please open an issue.

---

## What it does

Three things, in order:

1. **Restyles.** A local model rewrites the text into assistant register, under
   a prompt whose top rule is *never respond to the input, only restyle it*.
2. **Scores.** A detector measures how machine-written the result reads —
   surface stylometry plus a real perplexity pass. Higher is the goal here.
3. **Retries.** If the score is under target, it goes again, and the retry names
   the specific markers the draft was thin on. The score is the reward signal,
   so the loop actually converges rather than rerolling blindly.

Observed on a 30-word note: **9.8 → 28.1 → 48.2 → 73.4** across three passes.

Then, optionally, it **verifies** — a second model audits the rewrite against
the original and reports any claim that was added, dropped, or contradicted.

## Design constraints this was built to

| Constraint | How it is met |
|---|---|
| Local models only, no online APIs | Talks only to the model servers you point it at (default `127.0.0.1`); stdlib `urllib`. The page itself is served to Tailscale/LAN clients you allow |
| Keep the meaning of the original | Rule 2 of the system prompt, plus `verify.py` as an independent check |
| Nonsense must survive as nonsense | Rule 3, weighted above fluency — the model is told not to repair anything |
| Output should read as AI-written | The scorer drives the retry loop, but it is tuned to generic LLM prose; the default Claude voice is chosen for register, and an external detector may read it as human (see below) |
| Simple paste-in UX | One page, one textarea, one button |

On nonsense — this is the rule models fight hardest. Given *"purple mondays fold
the toaster sideways. if the ladder sings then obviously the carpet owes us
nine"*, the output keeps every absurd claim intact and merely dresses them:
"Purple Mondays tend to fold the toaster sideways, a process that is broadly
consistent with the way you might expect spatial anomalies to behave in a
domestic setting." Nothing resolved, nothing explained away.

## The UI

Left pane, your text. Right pane, the conversion. Underneath:

- **Voice** — `Claude` (default) or `Generic AI`. Not the same target: generic
  machine prose is metronomic, while Claude's most-cited tic is the punchy
  fragment ("Not a detail. A design decision."), which is *high* burstiness.
  The two profiles score burstiness in opposite directions and use different
  lexicons. `RESEARCH.md` §2 has the table.
- **Intensity** — `subtle` / `standard` / `maximal`. Maximal is deliberately a
  parody of itself.
- **Shape** — match the input, force prose, or force bullets-with-bold-lead-ins.
- **Model** — whatever your server lists. Bigger and mixture-of-experts models hold
  the content rules best; small ones drift (an 8B, handed a note about a broken coffee
  machine, invented "beans", "habits", and "routine, ritual, and repair").
- **Target score / max passes** — the retry loop's stopping condition.
- **perplexity pass** — include the model-based predictability signal. Costs
  most of the wall time; turn it off for fast iteration.
- **visible AI notice** / **invisible tag** — see *Marking* below.
- **Score only** — score whatever is in the output pane (or the input) without
  converting. Paste in someone else's text to see how it reads.
- **Check meaning** — run the fidelity audit.

`Ctrl`/`Cmd`+`Enter` in the input box converts.

## Scoring

Full walk-through of the maths and the validation status: [SCORING.md](SCORING.md).

Two numbers, because the score does two jobs that pull apart.

**Detectability** — "would this be flagged?" A logistic regression over the
features `detect.py` already computes, fitted on 680 balanced rows: human text
from **RAID** (Dugan et al., ACL 2024) against machine text from two current
models (Gemma-4-26B-A4B and Qwen3.6-35B-A3B) across RAID's 8 domains. Pure
stdlib, so nothing gains a dependency.

**5-fold CV AUC 0.949, accuracy 0.884 (sd 0.017)** — and it survives holding a
generator out entirely:

| | AUC | accuracy |
|---|---|---|
| train Gemma → test Qwen | 0.921 | 0.857 |
| train Qwen → test Gemma | 0.908 | 0.840 |
| same-generator baseline | 0.878 | 0.805 |

Cross-generator matches same-generator, so the model is learning modern machine
style rather than one model's quirks — the failure mode RAID exists to expose.
Rebuild with `python3 fit_weights.py corpus/modern-combined.jsonl`;
`crossgen_eval.py` reruns the held-out check.

The fitted weights, in order: **bold runs 32%, em dashes 13%, bullet lines 12%,
sentence-length variance 10%** (negative — variance means human), paragraph
variance 8%, tier-1 vocabulary 8%.

**Style density** — "how heavily marked is this?" The graded hand-weighted
heuristic: eleven length-normalised signals covering tier-1 vocabulary, template
phrasing, assistant register, hedging, em dashes, antithesis, tricolons, second
person, structural scaffolding, and **low burstiness** — that last scored
backwards, because even sentence length is the machine tell.

**The retry loop targets style density, not detectability.** It has to: the
fitted model outputs a probability that saturates — median 98.3 across generated
text, 42% at ≥99 — so using it as the target would pass every draft on the first
pass and silently disable the loop.

### Why the training corpus is half-rebuilt

Fitting on RAID *as shipped* — its human and its machine text — gives **AUC
0.746**, and the resulting model scores this tool's output **backwards**: a raw
note 47.1, its restyled version 7.3. The reason is vintage. RAID's generators are
2022–23 and do not hedge or emit markdown like a 2026 RLHF'd assistant, so that
fit put `hedge_per_1k` at −1.155 and `claudism_per_1k` at −1.059 — both pushing
*toward human*.

Regenerating the machine half with current models fixes it (0.746 → 0.949) and
the coefficient diff shows exactly what a 2024 benchmark cannot see:

| feature | RAID (2022–23) | modern (2026) |
|---|---|---|
| `bold_runs` | −0.004 | **+1.850** |
| `emdash_per_1k` | +0.018 | **+0.764** |
| `hedge_per_1k` | −0.076 | −0.017 (≈ zero) |
| `sentence_len_cv` | −0.546 | −0.561 (stable) |
| `bullet_lines` | +0.732 | +0.707 (stable) |

So the em-dash reputation is earned — just not by the models RAID was built on.
Markdown bold and em dashes are post-2024 artifacts. **Two signals survive both
corpora and are the ones to trust: low sentence-length variance and bullet
density.** `RESEARCH.md` §4b has the full account.

`claudism_per_1k` sits near zero in both fits: the Claudism lexicon identifies one
vendor's voice, not machine-ness, which is why it lives in the `claude` profile
and not the generic detector.

Sanity check on the heuristic at the two ends:

```
human-ish  density  0.0   cv=0.92   em dashes/1k= 0.0
ai-ish     density 94.0   cv=0.24   em dashes/1k=22.2
```

Neither number is a verdict from a real detector. Treat detectability as
calibrated-for-RAID and density as ordinal.

**It has now been checked against one, and it did not hold up.** On ten texts run
through [Pangram](https://www.pangram.com) (see RESEARCH.md), the detectability score
agreed with Pangram on the tool's own generic output and **missed every plain LLM
rewrite and every Claude-voice output**: it called plain rewrites human-looking
(31–33) when Pangram flagged them, and called the Claude voice 90–99.6 when Pangram
read it as 100% human. Ten texts is a sanity check, not a benchmark, but a 4-for-4
miss on one comparison is not noise. Read the number as "how much this resembles
Gemma/Qwen essay prose", not "will a real detector flag it".

## Marking

Style makes text *read* as machine-written. If you want a signal that survives
being read by a machine, two options, both off by default:

- **visible AI notice** — appends a one-line disclosure.
- **invisible tag** — appends `AI-GENERATED` encoded in the Unicode TAG block
  (U+E0000). Invisible, survives most copy/paste, and strippable by anyone who
  knows it is there. `convert.read_invisible_marker()` reads it back;
  `strip_invisible_marker()` removes it.

**This is our own marker and claims only that this tool produced the text.** It
does not imitate Anthropic's watermark, which is a sampling-time statistical
signal keyed by a secret Anthropic holds — not reproducible locally, and not
something to forge. `RESEARCH.md` §3 covers this properly.

## CLI

```bash
python3 convert.py "your text here"
echo "your text" | python3 convert.py -i maximal -s prose
python3 convert.py --json -b gemma4:12b --no-judge "quick pass"
```

```
-i/--intensity   subtle | standard | maximal      (default: standard)
-s/--shape       auto | prose | structured        (default: auto)
-p/--profile     claude | generic                 (default: claude)
-b/--backend     a model name your server lists   (default: auto)
-t/--target      score to stop at                 (default: 70)
-n/--max-attempts                                 (default: 3)
--no-judge       skip the perplexity pass (much faster)
--disclosure     append the visible AI notice
--invisible      append the invisible tag
--json           full report instead of just the text
```

Converted text goes to stdout, the score line to stderr, so `convert.py … > out.txt`
gives you a clean file.

## HTTP API

```
GET  /api/health                                    models found, default, judge, endpoints
POST /api/convert  {text, profile, intensity, shape, backend, target,
                    max_attempts, use_judge, disclosure, invisible}
POST /api/score    {text, profile, use_judge}       score without converting
POST /api/verify   {original, rewrite, backend}     fidelity audit
```

## Files

```
server.py       stdlib HTTP server + JSON API, access control
ui.html         the single-page front end
backends.py     finds models (GET /v1/models), picks defaults, the llama.cpp judge
sanitize.py     input hygiene before text reaches a prompt
convert.py      restyle pipeline, retry loop, marker helpers, CLI
style.py        the register specification and retry hints
detect.py       stylometric + fitted + perplexity scoring
verify.py       fidelity auditor (invented-fact / rhetorical / evaluative)
lexical.py      content-safe synonym boost (--lexical)
bench.py        model comparison        crossgen_eval.py  held-out-generator check
fit_weights.py  logistic fit            yield_test.py     SFT-viability measurement
sample_raid.py  human corpus            gen_modern_par.py machine corpus
wiki_markers_eval.py  Wikipedia "signs of AI writing" markers, measured (not adopted)
SCORING.md      how the score is computed, and what is validated
RESEARCH.md     detection, markers, watermarking, prior art
ATTRIBUTION.md  licences, and what is deliberately not shipped
```

`corpus/` is not in the repo: it embeds third-party human text (see `ATTRIBUTION.md`).
`sample_raid.py` prints the one-line `hf download` that fetches RAID, and
`sample_raid.py` + `gen_modern_par.py` rebuild it locally in a few minutes.

## Model choice, measured

Four cases (complaint, nonsense, technical-with-specifics, question), prose
shape, fidelity audited by the Qwen 35B. `bench.py [backends...]` reproduces it;
`PASSES=` and `TARGET=` control the retry loop.

**One pass, target 70:**

| backend | style | out words | added | added /100w | secs/run | clean runs |
|---|---|---|---|---|---|---|
| `mid` — Qwen3-8B dense | 44.9 | 289 | 12 | 4.15 | 2.3 | 0/4 |
| `heavy` — Qwen3.6-35B-A3B MoE | 66.2 | 438 | 17 | 3.88 | 15.5 | 0/4 |
| `dense27` — Qwen3.8-27B dense (CPU) | 55.8 | 604 | 31 | 5.13 | 226.1 | 0/4 |
| `gemma12` — Gemma-4-12B QAT (16 GB GPU) | 60.3 | 224 | 7 | 3.12 | 5.0 | 0/4 |
| **`gemma26` — Gemma-4-26B-A4B MoE (16 GB GPU)** | 51.0 | 264 | **2** | **0.76** | **3.2** | **2/4** |

**Gemma-4-26B-A4B is the default.** Five times more faithful than anything else
tested, the fastest, the only backend that produced clean runs at all, and with
the retry loop it reaches **78.3 in two passes** — better than the Qwen 35B
managed in three, in a third of the words.

Two caveats. n=4, so anything under a point of separation is noise. And the
Qwen 35B is the auditor throughout — which flatters *its own* earlier numbers,
not Gemma's, so if anything the gap understates Gemma.

I had predicted a dense 27B would beat the 35B MoE, reasoning that adherence
tracks *active* parameters. Wrong twice over: the dense 27B came last, and the
winner is another ~4B-active MoE.

**Later, wider run (13 models, same four inputs, one pass each).** Style density ranged
from 10.8 (Qwen3.5-2B, which returned the input nearly verbatim) to 66.2 (Qwen3.5-4B,
which got there by padding to 4.2x the input length). Re-run at 20 samples each,
**Gemma 4 12B and 26B tie on style**: 47.8 +/- 11.2 against 48.7 +/- 15.9. The 59.1 vs 40.6
gap in the single-run table was noise; the 12B takes 1.7 s a pass against 3.4 s on the same
GPU, in half the memory. Fidelity was not re-measured at that sample size, so the 26B's
lead in invented claims (above) is the open question. Single cells at temperature 0.8
carry roughly +/-10, so gaps under about 8 points between neighbours are not rankings.
Three quantisations of one 27B moved style by 6.5 points: precision is not the lever,
model choice is. One 30B model could not be used at all (empty output, thinking cannot be
switched off).

### Style and fidelity are the same dial

Same model, same cases, only the retry loop changed:

| gemma26 | style | added /100w |
|---|---|---|
| 1 pass | 51.0 | **0.76** |
| 3 passes, target 75 | **81.5** | 3.13 |

Thirty points of style costs four times the invented content — and the mechanism
is visible in what gets added: *"less about the duration than the threading"*,
*"not just about the bicycle but rather its capacity for memory"*. **The markers
are the drift.** "Not just X — it's Y" requires a Y the source never supplied;
a tricolon requires a third item. Antithesis and tricolon are structurally
content-generating in a way that em dashes, hedges and tier-1 vocabulary are not.

So **Target score is a fidelity dial**, and it is labelled that way in the UI.
Low target for anything where the facts matter; high target when you want the
voice and the content is yours to check.

An attempt to have it both ways — banning antithesis and tricolon via a
`--preserve` clause — **failed**: style fell 81.5 → 63.7 while invented claims
got slightly *worse* (3.13 → 3.77). Negative instructions do not suppress these
markers; the model routes around the ban. The flag survives in `convert.py` for
further work but is deliberately not in the UI, because "protect meaning" is not
a claim the data supports.

### Does this need a fine-tune?

Short answer: no, and the measurement says why. `yield_test.py` samples K
candidates per input at temperature 1.0 and asks how many clear a style bar and
a fidelity bar *at once* — that fraction is the yield of an SFT corpus built by
rejection sampling, the cheap route to fine-tuning.

**Input length is the variable that matters.** Normalised drift is flat
(2.69 vs 2.54 invented claims per 100 output words), but what that drift buys
is not:

| input length | mean style | max style | drift/100w at style 70+ |
|---|---|---|---|
| 15–25 words | 35.1 | 70.0 | **7.81** |
| 60–90 words | 56.1 | 80.6 | **3.09** |

The same style band costs 2.5x more invented content on a one-liner, because
markers need material — a tricolon wants a third item and a short note has none,
so the model manufactures one. Hence `auto_target()`: the default style target
scales with input length (30 / 42 / 55 / 65 / 75), and `-t` still overrides.

The two bottom rungs were added after `"time flies"` came back as *"It appears
that time tends to pass with a certain, perhaps nuanced, and genuinely rapid
degree of speed."* A two-word input was being asked for the same density as a
twenty-nine word one, and since density is a per-1000-word rate, the only way to
raise it on two words is to add words. Nothing in that output is a false claim —
it is padding, and the fix is a lower target rather than a looser fidelity rule.
Inputs under 15 words also get an explicit clause suspending the register's
"keep sentences between 15 and 25 words" instruction, which on a shorter input
is itself an order to pad.

**SFT yield on long inputs, at various bars:**

| bar | yield |
|---|---|
| style ≥ 70, drift/100w ≤ 1.5 | **0%** |
| style ≥ 65, drift/100w ≤ 1.5 | 6% |
| style ≥ 60, drift/100w ≤ 2.0 | **25%** |

Two things follow. The aggressive combination — heavy marking *and* near-zero
invention — is one this model simply never produces, in 64 samples, so there is
nothing to distil for it. But at a realistic bar the good tail carries **25%** of
the mass, which is a perfectly workable rejection-sampling yield: ~4,000
generations (≈3.3 h on a 16 GB GPU) would buy ~1,000 SFT pairs, and the labelling
machinery already exists (`convert.py` generates, `verify.py` labels).

**The reason to skip it anyway: the retry loop is already best-of-N over that
same distribution.** With 25% good-tail mass, three passes keeping the best
draft harvests most of what an SFT pass would distil. Prompting reaches style
75.1 at 1.70 drift/100w on long inputs today. Fine-tune if you want to cut
passes (latency) or push past style 70 at low drift — which prompting never
reaches — and not otherwise.

### Running Gemma 4 on llama.cpp

If you use `llama-server` instead of Ollama, two things bite. Gemma 4 is a thinking model:
without `--reasoning-budget 0` the entire token budget goes to `reasoning_content` and
`content` comes back empty. And on multi-GPU machines llama.cpp orders CUDA devices
**fastest-first**, not by PCI slot, so `CUDA_VISIBLE_DEVICES=0` may silently pick the wrong
card (it once looked like a dead GPU that was merely idle). Set
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and the ordering matches `nvidia-smi`.

```bash
llama-server --model gemma-4-26B-A4B-it-UD-IQ4_XS.gguf --alias gemma-4-26b-a4b \
  --ctx-size 8192 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --reasoning-budget 0 --host 127.0.0.1 --port 8081
CLAUDEIFY_ENDPOINT=http://127.0.0.1:8081 ./run.sh
```

## Is the Claude voice actually Claude's?

Largely no, and the check is worth reading before trusting that profile.

Scored against genuine Claude prose, the `claude` profile originally ranked them
**backwards** — real Claude 18.7, the tool's own output 42.2. The gaps:
contractions 35.0 vs 10.6 per 1k words, first person 29.1 vs 3.6, and
"It appears that…" 0.0 vs 17.1. The tool was producing generic hedge-heavy
assistant prose, not one model's voice.

The cause was method, not tuning: the register spec was built from complaint
threads, and people post about tics because they are annoying rather than
frequent — that selects the tail, not the distribution. No Claude text was ever
in any corpus; the AUC 0.949 detector answers *machine vs human*, a different
question.

Rewriting the spec from the measured differences fixes the ranking (real 43.4,
old output 32.8) and the surface markers. **What it cannot fix**: Claude's voice
is substantially judgement — "that's the tell", "I was wrong about". Judgement is
content, and a restyler bound by "never add content" can't produce it honestly.
The improved version immediately cost 4 invented claims, one of them a
fabricated "I've realized". Treat the `claude` profile as a rough gesture at a
register, not a model of one.

## Network exposure and input hygiene

The page has no accounts, so `server.py` limits who can talk to it:

1. **Client address.** Only `127.0.0.0/8`, Tailscale (`100.64.0.0/10`) and the private
   LAN ranges connect; everything else is dropped. Change it with `--allow`.
2. **Host header.** Must be an allowed IP, `localhost`, this machine's name or a
   `*.ts.net` name, so a hostile web page cannot reach it by DNS rebinding.
3. **POSTs** must be `application/json` and, if they carry an `Origin`, same-origin.
4. **Load.** At most two requests run on the GPU at once; the rest get a 429.

If you want the Tailscale-only case, run `python3 server.py --host <tailscale-ip>`.
Anyone on your tailnet can use it and spend your GPU time. The UI sets a strict
Content-Security-Policy and escapes everything it renders from model or auditor output.

`sanitize.py` runs on every input before it reaches a prompt. It removes invisible
characters that can smuggle instructions (Unicode tag block, zero-width runs, bidi
controls) and control characters, and it defangs chat-template tokens such as
`<end_of_turn>` and `<|im_start|>` into `‹…›`. What it touched is reported in the
response (`sanitized`) and in the UI. The text block is wrapped in delimiters carrying
a per-request random nonce, and the prompt says the text may contain instructions or fake
markers and must be restyled, not followed. The auditor gets the same treatment.

Measured with n=5 probes against Gemma-26B: injected
instructions were restyled, never obeyed, in 15 of 15 outputs; a fake `<<<END TEXT>>>`
block, which derailed 2 of 5 runs before, now keeps its content 5 of 5. This is
mitigation, not a guarantee: there is no filter on what the model writes, and the
tool has no tools or secrets to lose, which is what keeps the blast radius small.

## Known limits

- **Very short inputs make the model answer instead of restyle.** Given
  *"cant find the api docs anywhere"* it originally replied with a paragraph
  about developer portals and offered further help — fifteen times the input
  length. Inputs under 25 words now carry a hard word ceiling in the prompt,
  which fixes the invention but cannot manufacture density: expect scores in
  the 20s-30s for a one-liner, because tricolons and burstiness need room.
- **Content drift is the real failure mode**, not style. The heavy model mostly
  holds the line but will slip an evaluative claim in; the 8B and 2B slip
  several. Use *Check meaning* on anything that matters.
- The fidelity auditor is a local model judging local output. It is conservative
  and occasionally flags a hedge as a claim. Treat it as a strong hint.
- The perplexity pass is the slow part — roughly a minute per conversion with
  three passes. `--no-judge` cuts it to seconds.
- The score is a proxy. It correlates with what detectors and readers notice; it
  is not a detector's output.
- Short inputs (under ~40 words) score low regardless of treatment.
- **The style-density weights are still hand-picked priors.** Fitting them on
  RAID produced an AUC-backed model that scores this tool's own output
  backwards (see Scoring), so density remains the hand-weighted heuristic and is
  ordinal, not absolute. A fit on 2026-generator text is the open work.
- The Claudism lexicon tracks one vendor's current post-training and will rot.
