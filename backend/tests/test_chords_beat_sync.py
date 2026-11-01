"""extract_chords on synthetic audio: beat-synchronous matching must keep short chords.

Regression for chords lasting ~2 beats at ~117 BPM getting swallowed by the
frame path's 1s minimum segment.
"""

from __future__ import annotations

import numpy as np
import pytest

from chords import (
    _MIN_BEATS_FOR_SYNC,
    _MIN_CHORD_BEATS,
    _viterbi,
    _viterbi_min_duration,
    extract_chords,
)

SR = 22050
FREQS = {
    "D": [146.83, 293.66, 369.99, 440.00],
    "A": [110.00, 220.00, 277.18, 329.63],
    "Em": [164.81, 329.63, 392.00, 493.88],
    "G": [98.00, 196.00, 246.94, 293.66],
}
LABELS = list(FREQS)


def _render(progression):
    """[(label, seconds), ...] → mono float32 audio of sine-tone triads."""
    parts = []
    for label, secs in progression:
        t = np.arange(int(SR * secs)) / SR
        y = sum(np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) for f in FREQS[label])
        env = np.minimum(1, np.minimum(t / 0.01, (secs - t) / 0.03))
        parts.append(y * env)
    y = np.concatenate(parts).astype(np.float32)
    return y / (np.abs(y).max() * 1.1)


def _run(progression, beat_period, beats=True):
    y = _render(progression)
    duration = len(y) / SR
    beat_times = np.arange(beat_period, duration, beat_period) if beats else None
    segs = extract_chords(y, SR, duration, 0.0, allowed_chord_labels=LABELS, beat_times=beat_times)
    return [s for s in segs if s.label != "N"]


def test_two_beat_chords_all_detected():
    # 120 BPM, each chord held 2 beats (1.0s) — the case the 1s floor broke.
    prog = [("D", 1.0), ("A", 1.0), ("Em", 1.0), ("G", 1.0)] * 2
    segs = _run(prog, beat_period=0.5)
    assert [s.label for s in segs] == [label for label, _ in prog]
    for s in segs[1:-1]:
        assert s.end - s.start == pytest.approx(1.0, abs=0.15)


def test_one_beat_passing_note_is_absorbed():
    # A single beat of something else over a held chord is a passing note,
    # not a chord change, so the 2-beat minimum folds it into the D.
    prog = [("D", 2.0), ("A", 0.5), ("D", 2.0), ("G", 2.0)]
    segs = _run(prog, beat_period=0.5)
    assert [s.label for s in segs] == ["D", "G"]


def test_no_interior_chord_shorter_than_minimum():
    prog = [("D", 0.5), ("A", 0.5), ("Em", 1.0), ("G", 0.5), ("D", 1.5), ("A", 1.0)] * 2
    segs = _run(prog, beat_period=0.5)
    for s in segs[1:-1]:
        assert s.end - s.start >= _MIN_CHORD_BEATS * 0.5 - 0.1


def test_changes_land_on_beats():
    prog = [("D", 1.0), ("A", 1.0), ("Em", 1.0), ("G", 1.0)] * 2
    segs = _run(prog, beat_period=0.5)
    for s in segs[1:]:
        # Boundaries are quantized to ~93ms chroma frames, so allow one frame.
        assert min(abs(s.start - b) for b in np.arange(0.5, 8.0, 0.5)) < 0.1


class TestViterbiMinDuration:
    def test_min_len_1_matches_plain_viterbi(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            em = rng.standard_normal((6, 40))
            b = float(rng.uniform(0, 1))
            assert np.array_equal(_viterbi(em, b), _viterbi_min_duration(em, b, 1))

    def test_interior_runs_meet_minimum(self):
        rng = np.random.default_rng(1)
        for min_len in (2, 3):
            for _ in range(50):
                path = _viterbi_min_duration(rng.standard_normal((6, 40)), 0.1, min_len)
                runs = np.split(path, np.flatnonzero(np.diff(path)) + 1)
                assert all(len(r) >= min_len for r in runs[1:-1])

    def test_one_step_blip_is_suppressed(self):
        # State 1 wins a single step in the middle of a long run of state 0.
        em = np.zeros((2, 9))
        em[0, :] = 1.0
        em[1, 4] = 1.5
        assert _viterbi_min_duration(em, 0.0, 2).tolist() == [0] * 9


def test_falls_back_without_enough_beats():
    prog = [("D", 2.0), ("A", 2.0), ("Em", 2.0), ("G", 2.0)]
    y = _render(prog)
    duration = len(y) / SR
    few = np.arange(0.5, 0.5 * _MIN_BEATS_FOR_SYNC, 0.5)  # one short of the threshold
    for beat_times in (None, few):
        segs = extract_chords(y, SR, duration, 0.0, allowed_chord_labels=LABELS, beat_times=beat_times)
        assert [s.label for s in segs if s.label != "N"] == ["D", "A", "Em", "G"]
