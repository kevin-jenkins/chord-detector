"""Chord template library + music-theory constants.

Shared by chords.py (template matching), gemini_chords.py (sanitization +
key inference), and anywhere else that needs pitch-class math.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Flat → sharp equivalence map for normalizing chord roots (e.g. from Gemini).
_FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B",  "Fb": "E",   # rare but defensible
}

# Recognized quality suffixes — must match what _VOCAB_SETS["extended"] knows.
_KNOWN_QUALITIES = {"", "m", "7", "maj7", "m7", "sus2", "sus4", "dim", "aug"}

# Three vocabularies, in order of restrictiveness. Smaller vocabularies are
# more stable on noisy material (fewer confusable templates) but can only
# label chords approximately if the actual chord is a 7th/sus/etc.
_VOCAB_SETS = {
    "triads": {
        "":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "m":    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    },
    "triads_7ths": {
        "":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "m":    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "7":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "m7":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    },
    "extended": {
        "":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "m":    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "7":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "m7":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        "sus2": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "sus4": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        "dim":  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "aug":  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    },
}


def _center_and_normalize(matrix: np.ndarray, axis: int) -> np.ndarray:
    """Subtract mean along axis, then unit-normalize along the same axis.
    Result is suitable for Pearson correlation via matrix multiplication."""
    centered = matrix - matrix.mean(axis=axis, keepdims=True)
    norms = np.linalg.norm(centered, axis=axis, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return centered / norms


def _build_template_set(base_templates: dict) -> Tuple[List[str], np.ndarray]:
    """Return (labels, pearson-ready template matrix) for a quality set."""
    labels: List[str] = []
    rows = []
    for quality, base in base_templates.items():
        b = np.array(base, dtype=float)
        b /= np.linalg.norm(b)
        for root in range(12):
            rows.append(np.roll(b, root))
            labels.append(f"{PITCH_NAMES[root]}{quality}")
    matrix = np.array(rows)
    return labels, _center_and_normalize(matrix, axis=1)


# Pre-built template sets keyed by vocabulary name → (labels, pearson matrix)
_TEMPLATES: dict = {
    name: _build_template_set(base) for name, base in _VOCAB_SETS.items()
}

# Krumhansl-Schmuckler key profiles.
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
