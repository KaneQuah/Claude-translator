"""
lexical.py -- the safe half of the reversal.

"Humanizer" tools do post-hoc synonym substitution toward plain words
(utilize -> use, leverage -> use). Reversed, that is a word-level map toward
tier-1 assistant vocabulary.

Why this matters: it is the ONE marker-injection route that cannot invent
content. A tricolon needs a third item and an antithesis needs a second half --
both are propositions the source may not contain. A synonym swap changes which
word carries an existing proposition and nothing else. So it raises style
density at exactly zero factual risk, which is the coupling problem's only
clean escape.

Deliberately conservative: near-synonyms only, word-boundary matched, skipping
anything inside code spans or quotes.
"""

import re

# plain -> tier-1. Only near-synonyms; anything that shifts meaning is omitted.
SWAPS = {
    # Morphology matters: a naive uses?->leverages map produces "We leverages".
    r"\buses\b": "leverages", r"\buse\b": "leverage", r"\busing\b": "leveraging",
    r"\bshow\b": "underscore",
    r"\bimportant\b": "crucial", r"\bkey\b": "pivotal",
    r"\bcomplete\b": "comprehensive", r"\bthorough\b": "comprehensive",
    r"\bstrong\b": "robust", r"\bsturdy\b": "robust", r"\breliable\b": "robust",
    r"\bcomplicated\b": "nuanced", r"\bsubtle\b": "nuanced",
    r"\bwhole\b": "holistic", r"\boverall\b": "holistic",
    r"\bmany\b": "myriad", r"\blots of\b": "a myriad of",
    r"\bshows\b": "underscores", r"\bhighlights\b": "underscores",
    r"\bhandles\b": "navigates", r"\bhelps\b": "facilitates",
    r"\bdeal with\b": "navigate", r"\bhandle\b": "navigate",
    r"\bbig\b": "significant", r"\blarge\b": "substantial",
    r"\breal\b": "genuine", r"\breally\b": "genuinely",
    r"\bsmooth\b": "seamless", r"\beasy\b": "seamless",
    r"\bhelp\b": "facilitate", r"\bbuild\b": "foster",
    r"\bidea\b": "framework", r"\barea\b": "landscape",
    r"\busually\b": "generally", r"\boften\b": "frequently",
    r"\bmakes sure\b": "ensures", r"\bpoints to\b": "underscores",
}

_PROTECT = re.compile(r"(`[^`]*`|\"[^\"]*\"|“[^”]*”)")


def boost(text, limit=None):
    """
    Apply the substitutions outside code spans and quotations.

    `limit` caps how many swaps are made, so the caller can dial density
    without saturating. Returns (text, n_applied).
    """
    parts = _PROTECT.split(text or "")
    applied = 0
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:                      # protected span, pass through
            out.append(part)
            continue
        for pat, rep in SWAPS.items():
            if limit is not None and applied >= limit:
                break
            def _sub(m):
                nonlocal applied
                applied += 1
                w = m.group(0)
                # keep the original capitalisation
                return rep.capitalize() if w[:1].isupper() else rep
            part, n = re.subn(pat, _sub, part, count=(
                (limit - applied) if limit is not None else 0), flags=re.I)
        out.append(part)
    return "".join(out), applied
