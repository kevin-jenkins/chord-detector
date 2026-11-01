"""BTC (Bi-directional Transformer for Chord Recognition) — alternative local
chord recognizer.

The original BTC paper is Park & Choi, ISMIR 2019. The HuggingFace packaging
at puar-playground/btc-chord outputs Harte-notation segments natively,
including the 170-chord vocabulary (triads + 7ths + sus + dim + aug + inversions),
which is a strict superset of our _VOCAB_SETS["extended"]. Running it is one
line of transformers code once the dependency is installed.

Experimental. Install the extras first:

    pip install transformers huggingface_hub

===============================================================================
SECURITY: BTC inference uses `trust_remote_code=True`, which executes Python
from the puar-playground/btc-chord HuggingFace repo on every first load. If
that repo is compromised, flipping the BTC switch runs attacker code on your
machine. To make this an explicit opt-in, this module REFUSES to load the
model unless BTC_REVISION is set to a specific commit SHA. Pick one via:

    curl https://huggingface.co/api/models/puar-playground/btc-chord/refs

...then:

    export BTC_REVISION=<40-char sha>

Any time you want to upgrade, repeat the check and update the SHA. See the
"Experimental: swappable recognizers" section of the README.
===============================================================================

Enable via the per-request `chord_recognizer=btc` form field, or set the
env default with CHORD_RECOGNIZER=btc. Falls back to the stock Pearson +
Viterbi path on any failure (import error, model download fail, revision
unset, or empty output), so a missing install never breaks a request.

Notes:
- BTC produces its own time-aligned segments. We don't feed it through our
  Viterbi — BTC is the recognizer. Gemini's chord-set / key constraints
  used by the hybrid path do NOT apply when BTC is active.
- Harte qualities outside our vocabulary are degraded to the closest triad
  on the same root rather than dropped — otherwise long stretches of the
  song would show as "N". The map below covers the 170-class vocab.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from models import ChordSegment
from templates import _FLAT_TO_SHARP, _KNOWN_QUALITIES, PITCH_NAMES

_btc_model = None  # cached across requests
_BTC_REVISION = os.environ.get("BTC_REVISION", "").strip()
_BTC_REPO = "puar-playground/btc-chord"

# Harte "quality" → our shorthand. Keys compared against the colon-suffix
# part of a Harte label; values are our _KNOWN_QUALITIES vocab.
#
# The BTC 170-class vocab contains quite a bit more than our 9 qualities, so
# anything with a sensible closest-match degrades to it (an extended chord
# collapses to its triad on the same root; a half-dim or diminished-seventh
# collapses to dim; etc.). Unknown-prefix minor variants degrade to "m" via
# the fallthrough in _harte_to_shorthand.
_HARTE_QUALITY_MAP = {
    # Exact-match qualities we have templates for.
    "":        "",     # bare root = major
    "maj":     "",
    "min":     "m",
    "7":       "7",
    "maj7":    "maj7",
    "min7":    "m7",
    "sus2":    "sus2",
    "sus4":    "sus4",
    "dim":     "dim",
    "aug":     "aug",
    # Extended / enriched chords — degrade to the triad on the same root.
    "6":       "",
    "9":       "7",
    "11":      "7",
    "13":      "7",
    "maj6":    "",
    "maj9":    "maj7",
    "maj11":   "maj7",
    "maj13":   "maj7",
    "add9":    "",
    "add11":   "",
    "7sus4":   "sus4",
    "7sus2":   "sus2",
    "9sus4":   "sus4",
    # Minor family.
    "min6":    "m",
    "min9":    "m7",
    "min11":   "m7",
    "min13":   "m7",
    "minmaj7": "m",
    "minmaj9": "m",
    "min7b5":  "dim",
    # Diminished / augmented family.
    "dim7":    "dim",
    "hdim7":   "dim",
    "aug7":    "aug",
    "augmaj7": "aug",
    "aug9":    "aug",
    # Dominant alterations — fold back into the base 7.
    "7b5":     "7",
    "7#5":     "7",
    "7b9":     "7",
    "7#9":     "7",
    "7b13":    "7",
    "7#11":    "7",
}


def _harte_to_shorthand(label: str) -> Optional[str]:
    """Translate a Harte-notation label (e.g. "D:min", "F#:7/A#") to our
    shorthand ("Dm", "F#7"). Returns None for "N" (no chord) or anything
    we can't recognize.
    """
    if not label or label == "N":
        return None
    s = label.strip()

    # Strip bass note (slash); it doesn't change the chord template.
    if "/" in s:
        s = s.split("/", 1)[0].strip()

    # Pull apart "ROOT:QUALITY". Harte uses ":"; a bare root ("D") means major.
    if ":" in s:
        root_part, qual_part = s.split(":", 1)
    else:
        root_part, qual_part = s, ""

    m = re.match(r"^([A-G])([#b]?)$", root_part)
    if not m:
        return None
    root = _FLAT_TO_SHARP.get(root_part, root_part)
    if root not in PITCH_NAMES:
        return None

    qual = _HARTE_QUALITY_MAP.get(qual_part.strip())
    if qual is None:
        # Unknown Harte quality — degrade to major/minor triad by scanning
        # the leading tokens for a 'min' hint.
        lower = qual_part.lower()
        qual = "m" if lower.startswith("min") else ""
    if qual not in _KNOWN_QUALITIES:
        return None

    return f"{root}{qual}"


def _get_btc_model():
    """Lazy-load the BTC model once per process.

    Raises RuntimeError when BTC_REVISION is unset — see the module-level
    SECURITY block for the rationale. Caller is expected to catch and fall
    back silently.
    """
    global _btc_model
    if _btc_model is not None:
        return _btc_model
    if not _BTC_REVISION:
        raise RuntimeError(
            "BTC_REVISION env var not set. BTC uses trust_remote_code=True "
            "and loading an unpinned revision is unsafe. Set BTC_REVISION to "
            f"a specific commit SHA from https://huggingface.co/{_BTC_REPO}/commits"
        )
    from transformers import AutoModel
    print(
        f"Loading BTC chord recognizer from {_BTC_REPO}@{_BTC_REVISION[:12]} "
        f"(trust_remote_code=True — this executes third-party Python)..."
    )
    _btc_model = AutoModel.from_pretrained(
        _BTC_REPO,
        trust_remote_code=True,
        revision=_BTC_REVISION,
    )
    return _btc_model


def extract_chords_btc(audio_path: str, duration: float) -> List[ChordSegment]:
    """Run BTC on the given audio file and return merged chord segments.

    Raises on any failure — the caller is expected to catch and fall back.
    Returns an empty list only if the output is completely empty, which
    the caller should treat as a soft failure (not an exception).
    """
    model = _get_btc_model()
    raw = model.predict(audio_path)
    if not raw:
        return []

    # Normalize to list of (start, end, our-shorthand-label-or-None) tuples.
    normalized = []
    for item in raw:
        try:
            s = float(item["start"])
            e = float(item["end"])
            lab = str(item["label"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        shortcut = _harte_to_shorthand(lab)
        normalized.append((s, e, shortcut if shortcut is not None else "N"))

    if not normalized:
        return []

    # Clip to song bounds and merge adjacent same-label segments.
    merged: list = []
    for s, e, lab in normalized:
        s = max(0.0, min(s, duration))
        e = max(s, min(e, duration))
        if e <= s:
            continue
        if merged and merged[-1][2] == lab:
            ps, _, pl = merged[-1]
            merged[-1] = (ps, e, pl)
        else:
            merged.append((s, e, lab))

    return [ChordSegment(start=s, end=e, label=l) for s, e, l in merged]
