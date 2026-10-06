"""
sanitize.py -- input hygiene for text that is about to be placed in a prompt.

Stdlib only. The restyler has no tools and no secrets, so what an attacker can
reach is the integrity of the output and the shared GPU. These are the cheap
closes for the first of those:

  * invisible characters that carry instructions a human reviewer cannot see
    (Unicode tag block, zero-width runs, bidi controls, variation-selector
    supplement), and control characters that corrupt logs and terminals;
  * chat-template control tokens (<end_of_turn>, <|im_start|>, [INST] ...).
    llama-server does not tokenise these as special tokens from user content, but
    Gemma still half-reads them: in a probe, text after a fake <end_of_turn> was
    silently dropped 3 times in 5.

Everything removed or rewritten is counted, so the caller can say so rather
than change someone's text silently.
"""

import re
import unicodedata

# Always invisible, never needed in running prose.
_STRIP = re.compile(
    "[­​‎‏⁠-⁤⁦-⁩‪-‮﻿᠎"
    "\U000e0000-\U000e007f\U000e0100-\U000e01ef]")
# C0/C1 controls other than tab and newline.
_CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# ZWNJ / ZWJ are real characters in Persian, Indic scripts and emoji sequences,
# but also the usual carrier for binary-encoded hidden text. Keep one only when a
# non-ASCII character sits next to it.
_JOINER = re.compile("[‌‍]+")

_TOKENS = re.compile(
    r"<\|[^|<>\n]{1,40}\|>"                       # <|im_start|>, <|endoftext|>
    r"|</?(?:start_of_turn|end_of_turn|bos|eos|pad|unk|s|think)>"
    r"|\[/?INST\]|<<?/?SYS>>?",
    re.I)


def _defang(m):
    return "‹" + m.group(0)[1:-1] + "›"   # single angle quotes


def clean(text):
    """Return (cleaned_text, notes). `notes` maps what was touched to a count."""
    notes = {}
    text = text or ""

    def bump(key, n):
        if n:
            notes[key] = notes.get(key, 0) + n

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text, n = _STRIP.subn("", text)
    bump("invisible_chars", n)

    def joiner(m):
        s, e = m.start(), m.end()
        if e - s == 1:
            near = text[s - 1:s] + text[e:e + 1]
            if any(ord(c) > 127 for c in near):
                return m.group(0)
        bump("invisible_chars", e - s)
        return ""
    text = _JOINER.sub(joiner, text)

    text, n = _CTRL.subn("", text)
    bump("control_chars", n)

    text, n = _TOKENS.subn(_defang, text)
    bump("template_tokens", n)

    # Compatibility forms (fullwidth, ligatures) are left alone on purpose:
    # normalising them would change the author's characters, not remove hidden ones.
    return text.strip(), notes


def clean_input(text):
    return clean(text)[0]


def strip_markers(text):
    """Remove prompt delimiters a model may echo back into its output."""
    return re.sub(r"<<<(?:BEGIN|END) TEXT[^>\n]*>>>\s*", "", text or "").strip()
