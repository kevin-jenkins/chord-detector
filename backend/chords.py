"""Local chord recognition: chroma_cqt → Pearson template matching → Viterbi.

Pearson (not cosine) matters: with cosine similarity, noisy or near-uniform
chroma scores ~equally against every template, so C major wins by tiebreak.
Pearson centers the mean first, so uniform chroma scores ~0 against anything.

When beat times are available, matching is beat-synchronous: chroma is
summarized (median) within each beat and the Viterbi pass picks one chord per beat. The
shortest possible chord is one beat, and changes land on the beat grid. The
older frame-level path (~93ms frames, heavy smoothing, 1s minimum segment)
remains as a fallback for audio without a usable beat track. That 1s floor
swallowed 2-beat chords at ~117 BPM, which is why beat sync is the default.

The Viterbi pass biases toward self-transitions so two-guitar flicker (E ↔
Em) collapses — a competing chord has to beat the previous one by `self_bonus`
in Pearson score before the matcher flips.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

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


def _viterbi_min_duration(
    emission: np.ndarray, switch_penalty: float, min_len: int
) -> np.ndarray:
    """Like _viterbi, but once a state is entered it must last at least
    `min_len` steps. Switching costs `switch_penalty` (the same trade-off as
    _viterbi's self_bonus; min_len=1 reduces to it).

    Each state c expands into substates 0..min_len-1. (c,i) can only advance
    to (c,i+1); only the last substate can stay put or switch to another
    state's substate 0. The first and last segments are exempt: the path may
    start or end in any substate, since the song can begin or end mid-chord.
    """
    n_states, n_frames = emission.shape
    last = min_len - 1
    neg = -np.inf
    idx = np.arange(n_states)

    delta = np.full((n_states, min_len), neg)
    delta[:, 0] = emission[:, 0]
    delta[:, last] = emission[:, 0]
    back = np.zeros((n_frames, n_states, min_len, 2), dtype=np.int32)

    for t in range(1, n_frames):
        new = np.full((n_states, min_len), neg)
        bp = np.zeros((n_states, min_len, 2), dtype=np.int32)

        for i in range(1, min_len):  # advance within a chord
            new[:, i] = delta[:, i - 1]
            bp[:, i, 0], bp[:, i, 1] = idx, i - 1

        stay = delta[:, last]  # hold a chord that has met its minimum
        better = stay > new[:, last]
        new[better, last] = stay[better]
        bp[better, last, 0], bp[better, last, 1] = idx[better], last

        # Switch into substate 0 from the best *other* state's last substate.
        leave = delta[:, last] - switch_penalty
        order = np.argsort(leave)[::-1]
        best, second = order[0], order[1]
        src = np.where(idx == best, second, best)
        cand = leave[src]
        better = cand > new[:, 0]
        new[better, 0] = cand[better]
        bp[better, 0, 0], bp[better, 0, 1] = src[better], last

        delta = new + emission[:, t][:, None]
        back[t] = bp

    c, i = np.unravel_index(int(np.argmax(delta)), delta.shape)
    path = np.empty(n_frames, dtype=np.int32)
    for t in range(n_frames - 1, -1, -1):
        path[t] = c
        if t > 0:
            c, i = back[t, c, i]
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


# Any frame/beat where no chord template beats this Pearson floor routes
# through the N (no chord) state.
_N_FLOOR = 0.2
_CHROMA_HOP = 2048  # ~93ms per frame at sr=22050 (frame-level fallback)
# Finer frames for beat sync: with 93ms frames, beat boundaries snap to the
# wrong frame and each beat's chroma leaks into its neighbors, which halved
# the margin a one-beat passing chord needs to survive.
_BEAT_CHROMA_HOP = 512  # ~23ms
_MIN_BEATS_FOR_SYNC = 8
# Per-beat switch penalty, tuned together with _MIN_CHORD_BEATS on a 117 BPM
# track whose chords change every 2 beats. The minimum duration already
# suppresses flicker, so the penalty only needs to be small: 0.0-0.1 gave the
# same chart, and 0.2 started merging real 2-beat chords.
_BEAT_SELF_BONUS = 0.1
# Shortest chord the beat path will report. One-beat "changes" on real
# material turned out to be passing notes (bass or melody over a held chord),
# not chords; a 2-beat floor makes the matcher judge those beats together.
_MIN_CHORD_BEATS = 2


def _path_to_labels(path: np.ndarray, labels_for_state: List[str]) -> List[str]:
    n_state_idx = len(labels_for_state)  # last index = N
    return ["N" if i == n_state_idx else labels_for_state[i] for i in path.tolist()]


def _viterbi_labels(
    chroma: np.ndarray,
    template_z: np.ndarray,
    labels_for_state: List[str],
    self_bonus: float,
    min_len: int = 1,
) -> List[str]:
    sims = template_z @ _center_and_normalize(chroma, axis=0)  # (n_templates, n_cols)
    emission = np.vstack([sims, np.full((1, sims.shape[1]), _N_FLOOR)])
    if min_len > 1:
        path = _viterbi_min_duration(emission, switch_penalty=self_bonus, min_len=min_len)
    else:
        path = _viterbi(emission, self_bonus=self_bonus)
    return _path_to_labels(path, labels_for_state)


def _chroma(y_harm: np.ndarray, sr: int, tuning: float, hop: int) -> np.ndarray:
    return librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, hop_length=hop, bins_per_octave=36, tuning=tuning
    )


def _beat_sync_segments(
    y_harm: np.ndarray,
    sr: int,
    tuning: float,
    duration: float,
    beat_times: Sequence[float],
    template_z: np.ndarray,
    labels_for_state: List[str],
) -> List[tuple]:
    """One chord per beat: median chroma within each beat, Viterbi over beats."""
    chroma = _chroma(y_harm, sr, tuning, _BEAT_CHROMA_HOP)
    frames = librosa.time_to_frames(np.asarray(beat_times, dtype=float), sr=sr, hop_length=_BEAT_CHROMA_HOP)
    frames = np.unique(np.clip(frames, 0, chroma.shape[1] - 1))
    beat_chroma = librosa.util.sync(chroma, frames, aggregate=np.median)  # (12, len(frames)+1)
    edges = np.concatenate([
        [0.0], librosa.frames_to_time(frames, sr=sr, hop_length=_BEAT_CHROMA_HOP), [duration]
    ])
    labels = _viterbi_labels(
        beat_chroma, template_z, labels_for_state, _BEAT_SELF_BONUS, min_len=_MIN_CHORD_BEATS
    )

    segs: List[tuple] = []
    for i, label in enumerate(labels):
        start, end = float(edges[i]), float(min(edges[i + 1], duration))
        if end <= start:
            continue
        if segs and segs[-1][2] == label:
            segs[-1] = (segs[-1][0], end, label)
        else:
            segs.append((start, end, label))
    return segs


def _frame_segments(
    y_harm: np.ndarray,
    sr: int,
    tuning: float,
    duration: float,
    template_z: np.ndarray,
    labels_for_state: List[str],
) -> List[tuple]:
    """Frame-level fallback for audio without a usable beat track."""
    chroma = median_filter(_chroma(y_harm, sr, tuning, _CHROMA_HOP), size=(1, 5))  # ~465ms smoothing

    # 0.5 ≈ "a competing chord needs to score 0.5 higher in Pearson r to
    # overcome the cost of switching."
    labels = _viterbi_labels(chroma, template_z, labels_for_state, self_bonus=0.5)

    frame_times = librosa.frames_to_time(
        np.arange(len(labels)), sr=sr, hop_length=_CHROMA_HOP
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
    return merged


def extract_chords(
    y_harm: np.ndarray,
    sr: int,
    duration: float,
    tuning: float,
    vocabulary: str = "triads",
    allowed_chord_labels: Optional[List[str]] = None,
    beat_times: Optional[Sequence[float]] = None,
) -> List[ChordSegment]:
    """Chord recognition — see module docstring for pipeline rationale.

    When `allowed_chord_labels` is non-empty, templates are restricted to
    those chords (hybrid mode — Gemini identifies the chord set, we do the
    timing). Raises ValueError if fewer than 2 labels survive validation.
    With at least _MIN_BEATS_FOR_SYNC `beat_times` (seconds), matching is
    beat-synchronous; otherwise it falls back to the frame-level path.
    """
    if allowed_chord_labels:
        labels_for_state, template_z = _custom_templates_from_labels(allowed_chord_labels)
    else:
        if vocabulary not in _TEMPLATES:
            vocabulary = "triads"
        labels_for_state, template_z = _TEMPLATES[vocabulary]

    if beat_times is not None and len(beat_times) >= _MIN_BEATS_FOR_SYNC:
        segs = _beat_sync_segments(
            y_harm, sr, tuning, duration, beat_times, template_z, labels_for_state
        )
    else:
        segs = _frame_segments(y_harm, sr, tuning, duration, template_z, labels_for_state)

    return [ChordSegment(start=s, end=e, label=l) for s, e, l in segs]
