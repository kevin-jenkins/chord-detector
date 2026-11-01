"""Local chord recognition: chroma_cqt → Pearson template matching → Viterbi.

Pearson (not cosine) matters: with cosine similarity, noisy or near-uniform
chroma scores ~equally against every template, so C major wins by tiebreak.
Pearson centers the mean first, so uniform chroma scores ~0 against anything.

The Viterbi pass biases toward self-transitions so two-guitar flicker (E ↔
Em) collapses — a competing chord has to beat the previous one by `self_bonus`
in Pearson score before the matcher flips.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import librosa
import numpy as np
from scipy.ndimage import median_filter

from models import ChordSegment
from templates import (
    _FLAT_TO_SHARP,
    _TEMPLATES,
    _VOCAB_SETS,
    _center_and_normalize,
    MAJOR_PROFILE,
    MINOR_PROFILE,
    PITCH_NAMES,
)


def estimate_key(y_harm: np.ndarray, sr: int, tuning: float) -> str:
    """Krumhansl-Schmuckler key estimation on the harmonic component."""
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning)
    chroma_avg = chroma.mean(axis=1)

    best_score = -np.inf
    best_key = "C major"
    for i in range(12):
        major_score = np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_avg)[0, 1]
        minor_score = np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_avg)[0, 1]
        if major_score > best_score:
            best_score = major_score
            best_key = f"{PITCH_NAMES[i]} major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{PITCH_NAMES[i]} minor"
    return best_key


def _viterbi(emission: np.ndarray, self_bonus: float) -> np.ndarray:
    """Most-likely state path under a simple HMM where the only transition
    rule is "staying is cheaper than moving".

    emission: (n_states, n_frames) — per-state log-likelihood proxy
    Returns:  (n_frames,) — best state index per frame
    """
    n_states, n_frames = emission.shape
    delta = np.empty((n_frames, n_states))
    psi = np.empty((n_frames, n_states), dtype=np.int32)

    delta[0] = emission[:, 0]
    psi[0] = 0

    self_adj = np.eye(n_states) * self_bonus  # diagonal bonus, off-diagonal zero

    for t in range(1, n_frames):
        # scores[i, j] = delta[t-1, i] + (self_bonus if i == j else 0)
        scores = delta[t - 1, :, None] + self_adj
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores.max(axis=0) + emission[:, t]

    path = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def _custom_templates_from_labels(labels: List[str]) -> Tuple[List[str], np.ndarray]:
    """Build a Pearson-ready template matrix for an arbitrary list of chord
    labels (e.g. ["D", "A", "Em", "G"]). Labels whose quality we don't have
    a known bit-pattern for are silently dropped.

    Returns (kept_labels, centered_normalized_matrix). Raises ValueError if
    fewer than 2 labels survived filtering.
    """
    qualities = _VOCAB_SETS["extended"]
    rows: List[np.ndarray] = []
    kept: List[str] = []
    seen: set = set()

    for raw in labels:
        m = re.match(r"^([A-G])([#b]?)(.*)$", (raw or "").strip())
        if not m:
            continue
        root_letter, accidental, qual = m.group(1), m.group(2), m.group(3)
        root = _FLAT_TO_SHARP.get(root_letter + accidental, root_letter + accidental)
        if root not in PITCH_NAMES:
            continue
        if qual not in qualities:
            continue
        canonical = f"{root}{qual}"
        if canonical in seen:
            continue

        base = np.array(qualities[qual], dtype=float)
        base /= np.linalg.norm(base)
        rows.append(np.roll(base, PITCH_NAMES.index(root)))
        kept.append(canonical)
        seen.add(canonical)

    if len(kept) < 2:
        raise ValueError(
            f"only {len(kept)} valid template(s) from allowed_chord_labels"
        )

    matrix = np.array(rows)
    return kept, _center_and_normalize(matrix, axis=1)


def extract_chords(
    y_harm: np.ndarray,
    sr: int,
    duration: float,
    tuning: float,
    vocabulary: str = "triads",
    allowed_chord_labels: Optional[List[str]] = None,
) -> List[ChordSegment]:
    """Chord recognition — see module docstring for pipeline rationale.

    When `allowed_chord_labels` is non-empty, templates are restricted to
    those chords (hybrid mode — Gemini identifies the chord set, we do the
    timing). Raises ValueError if fewer than 2 labels survive validation.
    """
    if allowed_chord_labels:
        labels_for_state, template_z = _custom_templates_from_labels(allowed_chord_labels)
    else:
        if vocabulary not in _TEMPLATES:
            vocabulary = "triads"
        labels_for_state, template_z = _TEMPLATES[vocabulary]

    hop = 2048  # ~93ms per frame at sr=22050
    chroma = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, hop_length=hop, bins_per_octave=36, tuning=tuning
    )
    chroma = median_filter(chroma, size=(1, 5))  # ~465ms smoothing

    chroma_z = _center_and_normalize(chroma, axis=0)        # (12, n_frames)
    sims = template_z @ chroma_z                            # (n_templates, n_frames)

    # Treat the silence threshold itself as a state: any frame where no chord
    # template beats this floor will route through the N state.
    N_FLOOR = 0.2
    n_emission = np.full((1, sims.shape[1]), N_FLOOR)
    emission = np.vstack([sims, n_emission])

    # 0.5 ≈ "a competing chord needs to score 0.5 higher in Pearson r to
    # overcome the cost of switching." Tuned for rhythm-guitar/bass-anchored
    # detection — the matcher only flips on actual chord changes.
    path = _viterbi(emission, self_bonus=0.5)

    n_state_idx = len(labels_for_state)  # last index = N
    labels = [
        "N" if i == n_state_idx else labels_for_state[i]
        for i in path.tolist()
    ]

    frame_times = librosa.frames_to_time(
        np.arange(len(labels)), sr=sr, hop_length=hop
    )

    # Build raw segments at label-change boundaries.
    raw: List[tuple] = []
    seg_start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            raw.append((float(frame_times[seg_start]), float(frame_times[i]), labels[i - 1]))
            seg_start = i
    raw.append((float(frame_times[seg_start]), duration, labels[-1] if labels else "N"))

    # Single-pass merge: absorb short segments into predecessor and merge
    # adjacent segments sharing the same label.
    MIN_DUR = 1.0
    merged: List[tuple] = []
    for start, end, label in raw:
        if not merged:
            merged.append((start, end, label))
            continue
        prev_start, prev_end, prev_label = merged[-1]
        if prev_label == label or (end - start) < MIN_DUR:
            merged[-1] = (prev_start, end, prev_label)
        else:
            merged.append((start, end, label))

    return [ChordSegment(start=s, end=e, label=l) for s, e, l in merged]
