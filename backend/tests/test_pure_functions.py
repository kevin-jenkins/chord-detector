"""Tests for the pure functions in main.py.

These exist to catch regressions on numpy/librosa version bumps and on the
chord-vocabulary data structures. Nothing in here touches the network, the
file system, or the audio pipeline — all inputs are synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from main import (
    ChordSegment,
    _custom_templates_from_labels,
    _diatonic_chords,
    _infer_better_key,
    _sanitize_chord_label,
    _viterbi,
)


# ---- _sanitize_chord_label ------------------------------------------------

class TestSanitizeChordLabel:
    @pytest.mark.parametrize("raw, expected", [
        ("D", "D"),
        ("Dm", "Dm"),
        ("F#m", "F#m"),
        ("Bb", "A#"),          # flat → sharp normalization
        ("Ebm", "D#m"),
        ("G7", "G7"),
        ("Cmaj7", "Cmaj7"),
        ("Am7", "Am7"),
        ("Dsus4", "Dsus4"),
        ("Dsus2", "Dsus2"),
        ("Fdim", "Fdim"),
        ("Caug", "Caug"),
        ("D/F#", "D"),         # slash bass stripped
        ("  Dm  ", "Dm"),      # whitespace tolerant
    ])
    def test_known_shapes(self, raw, expected):
        assert _sanitize_chord_label(raw) == expected

    @pytest.mark.parametrize("bad", [
        "", "   ", "H", "D#foo", "Zm", None, 42,
    ])
    def test_unparseable_returns_none(self, bad):
        assert _sanitize_chord_label(bad) is None


# ---- _diatonic_chords -----------------------------------------------------

class TestDiatonicChords:
    def test_d_major(self):
        assert _diatonic_chords("D major") == ["D", "Em", "F#m", "G", "A", "Bm"]

    def test_e_minor(self):
        assert _diatonic_chords("E minor") == ["Em", "G", "Am", "Bm", "C", "D"]

    def test_c_major(self):
        assert _diatonic_chords("C major") == ["C", "Dm", "Em", "F", "G", "Am"]

    def test_flat_root_normalizes(self):
        # Bb major → A# major in internal shorthand
        chords = _diatonic_chords("Bb major")
        assert chords[0] == "A#"
        assert len(chords) == 6

    @pytest.mark.parametrize("bad", ["", "nonsense", "D", "D dorian", "Hm"])
    def test_bad_input_returns_empty(self, bad):
        assert _diatonic_chords(bad) == []


# ---- _infer_better_key ----------------------------------------------------

def _segs(labels_and_durations):
    """Build ChordSegment list from [(label, duration_seconds), ...]."""
    t = 0.0
    out = []
    for label, dur in labels_and_durations:
        out.append(ChordSegment(start=t, end=t + dur, label=label))
        t += dur
    return out


class TestInferBetterKey:
    def test_agrees_with_reported_passes_through(self):
        segs = _segs([("D", 10), ("A", 5), ("Em", 3), ("G", 4), ("D", 5)])
        key, prev = _infer_better_key("D major", segs)
        assert key == "D major"
        assert prev is None

    def test_strong_signal_overrides_wrong_mode(self):
        # First + last on D, song reported as E minor. Override expected.
        segs = _segs([("D", 10), ("A", 5), ("Em", 2), ("G", 4), ("D", 10)])
        key, prev = _infer_better_key("E minor", segs)
        assert key.startswith("D ")
        assert prev == "E minor"

    def test_weak_signal_does_not_override(self):
        # First=D, last=A, most-played=Em — total score 3.5 on Em, below 4.0.
        segs = _segs([("D", 2), ("Em", 20), ("G", 1), ("A", 3)])
        key, _prev = _infer_better_key("E minor", segs)
        assert key == "E minor"

    def test_too_few_segments_returns_reported(self):
        segs = _segs([("D", 10), ("A", 10)])  # fewer than 3 real segments
        assert _infer_better_key("D major", segs) == ("D major", None)

    def test_filters_short_and_N_segments(self):
        segs = _segs([
            ("D", 10), ("N", 20), ("A", 0.1), ("Em", 0.1), ("G", 0.1), ("D", 10)
        ])
        # Only D and nothing else is long enough — falls through to reported.
        key, _prev = _infer_better_key("D major", segs)
        assert key == "D major"


# ---- _viterbi -------------------------------------------------------------

class TestViterbi:
    def test_single_dominant_state_wins_everywhere(self):
        n_states, n_frames = 4, 10
        emission = np.full((n_states, n_frames), -1.0)
        emission[2, :] = 1.0  # state 2 always best
        path = _viterbi(emission, self_bonus=0.5)
        assert path.shape == (n_frames,)
        assert np.all(path == 2)

    def test_self_bonus_suppresses_one_frame_flicker(self):
        # Three frames: state 0 strongly favored at t=0 and t=2, state 1 slightly
        # favored at t=1. With no self-bonus, the path flickers 0→1→0; with a
        # strong self-bonus, it should stay on 0.
        emission = np.array([
            [5.0, -0.1, 5.0],  # state 0
            [0.0,  0.1, 0.0],  # state 1
        ])
        no_bonus = _viterbi(emission, self_bonus=0.0)
        with_bonus = _viterbi(emission, self_bonus=10.0)
        assert no_bonus.tolist() == [0, 1, 0]
        assert with_bonus.tolist() == [0, 0, 0]

    def test_output_dtype_and_shape(self):
        emission = np.random.default_rng(0).standard_normal((5, 7))
        path = _viterbi(emission, self_bonus=0.3)
        assert path.shape == (7,)
        assert path.dtype == np.int32
        assert path.min() >= 0 and path.max() < 5


# ---- _custom_templates_from_labels ----------------------------------------

class TestCustomTemplatesFromLabels:
    def test_basic_quartet(self):
        labels, mat = _custom_templates_from_labels(["D", "A", "Em", "G"])
        assert labels == ["D", "A", "Em", "G"]
        assert mat.shape == (4, 12)

    def test_flat_normalized_to_sharp(self):
        labels, _ = _custom_templates_from_labels(["Bb", "F"])
        assert labels == ["A#", "F"]

    def test_dedupes_canonical_form(self):
        # Bb == A#, so this should collapse to one.
        with pytest.raises(ValueError):
            _custom_templates_from_labels(["Bb", "A#"])

    def test_unknown_quality_dropped(self):
        # "sus9" isn't in our vocabulary; survives only the known ones.
        labels, _ = _custom_templates_from_labels(["D", "Asus9", "G"])
        assert labels == ["D", "G"]

    def test_rejects_single_label(self):
        with pytest.raises(ValueError):
            _custom_templates_from_labels(["D"])

    def test_rejects_all_garbage(self):
        with pytest.raises(ValueError):
            _custom_templates_from_labels(["Hfoo", "Zbar"])
