"""BTC (Bi-directional Transformer for Chord Recognition) — alternative local
chord recognizer.

The original BTC paper is Park & Choi, ISMIR 2019. The HuggingFace packaging
at puar-playground/btc-chord outputs Harte-notation segments natively,
including the 170-chord vocabulary (triads + 7ths + sus + dim + aug + inversions),
which is a strict superset of our _VOCAB_SETS["extended"]. Running it is one
line of transformers code once the dependency is installed.

Experimental. Install the extras first:

    pip install transformers huggingface_hub

Enable via the per-request `chord_recognizer=btc` form field, or set the
env default with CHORD_RECOGNIZER=btc. Falls back to the stock Pearson +
Viterbi path on any failure (import error, model download fail, or empty
output), so a missing install never breaks a request.

Notes:
- BTC produces its own time-aligned segments. We don't feed it through our
  Viterbi — BTC is the recognizer. Gemini's chord-set / key constraints
  used by the hybrid path do NOT apply when BTC is active.
- Harte qualities outside our vocabulary (e.g., min11, 9) are degraded to
  the closest triad on the same root rather than dropped — otherwise long
  stretches of the song would show as "N".
"""

from __future__ import annotations

import re
from typing import List, Optional

from models import ChordSegment
from templates import _FLAT_TO_SHARP, _KNOWN_QUALITIES, PITCH_NAMES

_btc_model = None  # cached across requests

# Harte "quality" → our shorthand. Keys compared against the colon-suffix
# part of a Harte label; values are our _KNOWN_QUALITIES vocab.
_HARTE_QUALITY_MAP = {
    "": "",         # bare root = major
    "maj": "",
    "min": "m",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "sus2": "sus2",
    "sus4": "sus4",
    "dim": "dim",
    "aug": "aug",
    # Degraded mappings — pick the closest triad on the same root.
    "min9": "m",
    "min11": "m",
    "9": "7",
    "11": "7",
    "13": "7",
    "6": "",
    "min6": "m",
    "dim7": "dim",
    "hdim7": "dim",
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
    """Lazy-load the BTC model once per process."""
    global _btc_model
    if _btc_model is not None:
        return _btc_model
    from transformers import AutoModel
    print("Loading BTC chord recognizer (first run downloads weights)...")
    _btc_model = AutoModel.from_pretrained(
        "puar-playground/btc-chord",
        trust_remote_code=True,
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
