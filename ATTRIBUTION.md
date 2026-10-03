# Attribution and third-party terms

The code in this repository is MIT (see `LICENSE`). What follows is what it
depends on, what it was built from, and — importantly — **what is deliberately
not shipped with it.**

## What is NOT in this repository, and why

`corpus/` is git-ignored. `sample_raid.py` and `gen_modern_par.py` rebuild it
locally in a few minutes.

The training corpus embeds **1,200 verbatim human texts** that RAID aggregated
from Reddit, IMDb-style reviews, arXiv abstracts, news, poetry, Wikipedia,
recipes and books. RAID itself is MIT, and that MIT grant covers *RAID's* own
contribution — the compilation, the generations, the code. **It cannot
relicense the underlying third-party content**, and the news, book and poetry
excerpts in particular are other people's copyrighted work.

So: redistributing the fitted **coefficients** (18 floats in
`fitted-weights.json`) is fine. Redistributing the **corpus they were fitted on**
is not something this project is in a position to authorise. Anyone who wants
the corpus should get RAID from the source.

## Upstream

| What | Where | License |
|---|---|---|
| RAID benchmark (human text, the fitting corpus) | Dugan et al., ACL 2024 — `liamdugan/raid` | MIT |
| `avoid-ai-writing` (the P0/P1/P2 marker taxonomy, inverted here) | `conorbronsdon/avoid-ai-writing` | MIT |
| *Wikipedia:Signs of AI writing* (marker catalogue tested in `wiki_markers_eval.py`) | en.wikipedia.org | CC BY-SA 4.0 |
| llama.cpp (the inference server this talks to) | `ggml-org/llama.cpp` | MIT |
| Gemma 4 (default restyle backend) | `google/gemma-4-26B-A4B-it` | Apache-2.0 |
| Qwen 3.6 (fidelity auditor) | `Qwen/Qwen3.6-35B-A3B` | Apache-2.0 |

Both model families are Apache-2.0, so there is no custom-terms or
acceptable-use rider to pass on to users — a change from Gemma 3, which shipped
under Google's own Gemma Terms of Use. Model weights are never bundled here; the
tool talks to a server you run.

The marker phrase lists in `wiki_markers_eval.py` are drawn from the Wikipedia page
above. They are observations about word choice, not its prose, but the editors who
compiled them did the work, and the page is CC BY-SA, so it is credited here.
None of them is used by the scorer; see RESEARCH.md for why.

The Claude-tic lexicon in `detect.py` was compiled from public discussion
(r/ClaudeAI, r/ClaudeCode, Hacker News). Word lists of this kind are facts
rather than creative expression, but the observation is not mine and the
threads deserve the credit.

Cite RAID if you use the fitted model in anything published:

> Dugan et al. *RAID: A Shared Benchmark for Robust Evaluation of
> Machine-Generated Text Detectors.* ACL 2024.

## Intent

This tool makes text read as **more** machine-written, and reports how
detectable the result is. It is the inverse of the "humanizer" tools that strip
those markers to defeat detectors, and it is not built to help anyone pass work
off as their own. `--disclosure` and `--invisible` add AI-provenance markers
rather than remove them.

One asymmetry worth naming: text that *looks* AI-written is not evidence that it
*is*. Do not use this to make someone else's writing look machine-generated.
