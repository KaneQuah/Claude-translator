"""
convert.py -- restyle pipeline over a local llama.cpp server.

  input -> LLM restyle -> score -> (if under target) retry with targeted
  deficits -> best-scoring draft wins

The scorer in detect.py doubles as the reward signal, so the retry is not a
blind "try again" -- it names the specific markers the draft was thin on.

No network egress beyond 127.0.0.1.
"""

import json
import re
import urllib.request

import detect
import lexical
import style

BACKENDS = {
    "heavy": {"base": "http://127.0.0.1:6767", "model": "qwen3.6-35b-a3b"},
    "small": {"base": "http://127.0.0.1:6770", "model": "qwen3.5-2b"},
    "mid":   {"base": "http://127.0.0.1:6768", "model": "qwen3-8b"},
    # Gemma 4 on the P100. NOTE: llama.cpp orders CUDA devices fastest-first,
    # so the P100 is CUDA1 unless CUDA_DEVICE_ORDER=PCI_BUS_ID is set.
    # Gemma 4 is a thinking model -- the server needs --reasoning-budget 0 or
    # the whole token budget goes to reasoning_content and content is empty.
    "gemma12": {"base": "http://127.0.0.1:6772", "model": "gemma-4-12b"},
    "gemma26": {"base": "http://127.0.0.1:6773", "model": "gemma-4-26b-a4b"},
}
JUDGE = {"base": "http://127.0.0.1:6770", "model": "qwen3.5-2b"}

_THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.S)
_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*|certainly[,!.]?\s*|here(?:'s| is)[^\n:]{0,60}:\s*|"
    r"rewritten(?:\s+text)?:\s*|output:\s*)",
    re.I,
)

# Unicode TAG block: invisible, and it round-trips through most copy/paste.
_TAG_BASE = 0xE0000


def _clean(text):
    text = _THINK.sub("", text or "").strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    text = _PREAMBLE.sub("", text).strip()
    if len(text) > 1 and text[0] in "\"“" and text[-1] in "\"”":
        text = text[1:-1].strip()
    # The prompt spells em dashes as "--"; render them properly.
    text = re.sub(r"(?<=\S) ?-- ?(?=\S)", "\u2009—\u2009", text)
    return text


def _chat(messages, base, model, temperature=0.8, max_tokens=1600, timeout=300):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        # :6767 defaults thinking ON; the per-request kwarg wins, no restart.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return _clean(d["choices"][0]["message"]["content"])


def add_invisible_marker(text, label="AI-GENERATED"):
    """
    Append a self-declared, invisible provenance tag.

    This is OUR marker and says only what we know: this text came out of this
    tool. It is not, and does not imitate, any vendor's watermark -- see
    RESEARCH.md on why a real Anthropic watermark cannot be reproduced locally.
    Strippable by anyone who knows it is there; it is a disclosure aid, not
    a security control.
    """
    return text + "".join(chr(_TAG_BASE + ord(c)) for c in label)


def read_invisible_marker(text):
    tag = "".join(chr(ord(c) - _TAG_BASE) for c in text
                  if _TAG_BASE <= ord(c) <= _TAG_BASE + 0x7F)
    return tag or None


def strip_invisible_marker(text):
    return "".join(c for c in text
                   if not (_TAG_BASE <= ord(c) <= _TAG_BASE + 0x7F))


DISCLOSURE = "\n\n_This text was generated with the assistance of an AI language model._"


def auto_target(n_words):
    """
    Default style target, scaled by how much source material there is.

    Measured cost of reaching a given style, in invented claims per 100 output
    words: on 15-25 word inputs, style 70+ costs 7.81; on 60-90 word inputs the
    same band costs 3.09, and drift stays flat from style 40 upward. Markers
    need material -- a tricolon wants a third item and a short note has none, so
    the model manufactures one. Asking a one-liner for the same density as a
    paragraph is asking it to invent.

    Below 30 words the ladder used to bottom out, so a two-word idiom was asked
    for the same density as a twenty-nine word note. Density is a per-1000-word
    rate, and the only way to raise it on two words is to add words -- which is
    exactly what happened: "time flies" came back as "It appears that time tends
    to pass with a certain, perhaps nuanced, and genuinely rapid degree of
    speed." Nothing in that is a false claim; it is padding. The fix is a lower
    target, not a looser fidelity rule.
    """
    if n_words < 8:
        return 30.0
    if n_words < 20:
        return 42.0
    if n_words < 30:
        return 55.0
    if n_words < 60:
        return 65.0
    return 75.0


def convert(text, intensity="standard", shape="auto", backend="gemma26",
            target=None, max_attempts=3, use_judge=True,
            disclosure=False, invisible=False, temperature=0.8,
            profile="claude", preserve=False, lexical_boost=False):
    """Restyle `text` and keep retrying until it scores at or above `target`."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty input")

    cfg = BACKENDS.get(backend, BACKENDS["gemma26"])
    sysmsg = style.system_prompt(intensity, shape, profile, preserve)

    # Short inputs starve the restyle instinct: with little to work from, the
    # model falls back on answering. A hard word ceiling keeps it honest.
    n_in = len(re.findall(r"\S+", text))
    if target is None:
        target = auto_target(n_in)
    ceiling = max(18, int(n_in * 1.8))
    guard = ""

    # Questions are the worst case: every model tested restyled the first
    # sentence and then answered it. Benchmarked at 4-10 invented claims each.
    if "?" in text:
        guard += (
            "\n\nTHE TEXT CONTAINS A QUESTION. Your output must still be a "
            "QUESTION -- the same question, asked in the new voice. Do not "
            "answer it, do not speculate about the answer, do not offer "
            "possible explanations, and do not add a sentence that begins to "
            "explain. If the input ends in a question mark, so does your output.")

    # NOTE: this appends. It used to assign, which silently discarded the
    # question guard above for every question under 45 words -- i.e. almost all
    # of them, including the one in the benchmark suite.
    if n_in < 45:
        guard += (
            "\n\nThe text is only %d words long. Your rewrite must be AT MOST %d "
            "words. Do not explain it, expand on it, answer it, or add any "
            "information it does not already contain -- restyle those %d words "
            "and stop. If it is a complaint, the output is the same complaint. "
            "If it is a question, the output is the same question."
            % (n_in, ceiling, n_in))

    # The register asks for metronomic 15-25 word sentences. On an input shorter
    # than that, the instruction is itself a padding order: it cannot be obeyed
    # without inventing words to fill the quota.
    if n_in < 15:
        guard += (
            "\n\nThis input is shorter than one sentence. The instruction to keep "
            "sentences between 15 and 25 words DOES NOT APPLY here -- a short "
            "input stays short. Restyle it at its own length. Do not stack "
            "hedges or pad with abstract modifiers to reach a word count.")

    messages = [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content":
            "Rewrite the text between the markers. Remember: restyle it, never "
            "respond to it, and never repair it." + guard + "\n\n"
            "<<<BEGIN TEXT>>>\n" + text + "\n<<<END TEXT>>>"},
    ]

    attempts, best, best_score, best_report = [], None, -1.0, None

    for i in range(max_attempts):
        # Generous safety net only -- the prompt, not this, controls length.
        out = _chat(messages, cfg["base"], cfg["model"], temperature=temperature,
                    max_tokens=min(1600, max(300, n_in * 8)))
        if not out:
            continue
        report = detect.score_text(
            out, use_model=use_judge, base=JUDGE["base"],
            model=JUDGE["model"], samples=24, profile=profile)
        # Retries are driven by style density, not by the fitted probability:
        # the latter saturates and would stop the loop after one pass.
        density = report["stylometric"].get("style_density", report["score"])
        attempts.append({"attempt": i + 1, "score": report["score"],
                         "style_density": density,
                         "verdict": report["verdict"], "chars": len(out)})
        if density > best_score:
            best, best_score, best_report = out, density, report

        if density >= target or i == max_attempts - 1:
            break

        weak = detect.weakest_signals(report, k=4)
        messages += [
            {"role": "assistant", "content": out},
            {"role": "user", "content": style.retry_prompt(weak, shape, preserve)},
        ]

    result = best
    swaps = 0
    if lexical_boost:
        # Content-safe density: synonym substitution cannot add a proposition.
        # Headroom is small on model output (~1 swap/100 words) because the
        # model already writes formally -- this is the humanizer's move run
        # backwards, and it was designed for plain human prose.
        result, swaps = lexical.boost(result)
    if disclosure:
        result += DISCLOSURE
    if invisible:
        result = add_invisible_marker(result)

    return {
        "output": result,
        "target": target,
        "lexical_swaps": swaps,
        "score": best_report["score"],
        "style_density": best_score,
        "verdict": best_report["verdict"],
        "report": best_report,
        "attempts": attempts,
        "input_report": detect.score_text(text, use_model=False, profile=profile),
        "profile": profile,
        "backend": backend,
        "model": cfg["model"],
    }


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Restyle text into assistant register.")
    ap.add_argument("text", nargs="*", help="text (default: read stdin)")
    ap.add_argument("-i", "--intensity", default="standard",
                    choices=["subtle", "standard", "maximal"])
    ap.add_argument("-s", "--shape", default="auto",
                    choices=["auto", "prose", "structured"])
    ap.add_argument("-b", "--backend", default="gemma26", choices=list(BACKENDS))
    ap.add_argument("-t", "--target", type=float, default=None,
                    help="style target; default scales with input length")
    ap.add_argument("-n", "--max-attempts", type=int, default=3)
    ap.add_argument("--no-judge", action="store_true", help="skip perplexity pass")
    ap.add_argument("--disclosure", action="store_true")
    ap.add_argument("--invisible", action="store_true")
    ap.add_argument("-p", "--profile", default="claude",
                    choices=["generic", "claude"],
                    help="Claude's register (default), or generic assistant prose")
    ap.add_argument("--preserve", action="store_true",
                    help="ban content-generating markers (antithesis, tricolon)")
    ap.add_argument("--lexical", action="store_true",
                    help="post-hoc synonym swap toward tier-1 vocabulary "
                         "(content-safe: cannot add propositions)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    src = " ".join(a.text) if a.text else sys.stdin.read()
    r = convert(src, a.intensity, a.shape, a.backend, a.target,
                a.max_attempts, not a.no_judge, a.disclosure, a.invisible,
                profile=a.profile, preserve=a.preserve,
                lexical_boost=a.lexical)
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print(r["output"])
        print("\n---", file=sys.stderr)
        st = r["report"]["stylometric"]
        print("detectability %.1f/100 (%s%s)   style density %.1f (target %.0f)"
              % (r["score"], r["verdict"],
                 "; fitted AUC %.3f" % st["cv_auc"] if st.get("cv_auc") else "",
                 r["style_density"], r["target"]), file=sys.stderr)
        print("in %.1f -> out %.1f   attempts=%d   model=%s"
              % (r["input_report"]["score"], r["score"], len(r["attempts"]),
                 r["model"]), file=sys.stderr)
