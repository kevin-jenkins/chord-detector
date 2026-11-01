"""Tests for the Harte→shorthand conversion in btc.py.

BTC inference itself is not tested here — it needs a model download and real
audio. The pure parsing logic is the part that can regress silently.
"""

from __future__ import annotations

import pytest

from btc import _harte_to_shorthand


class TestHarteToShorthand:
    @pytest.mark.parametrize("raw, expected", [
        ("D", "D"),            # bare root = major
        ("D:maj", "D"),
        ("D:min", "Dm"),
        ("F#:7", "F#7"),
        ("C:maj7", "Cmaj7"),
        ("A:min7", "Am7"),
        ("D:sus2", "Dsus2"),
        ("D:sus4", "Dsus4"),
        ("F:dim", "Fdim"),
        ("C:aug", "Caug"),
        ("Bb:min", "A#m"),     # flat → sharp
        ("D:min/F", "Dm"),     # slash bass stripped
        ("D:min11", "Dm"),     # unknown quality degrades to triad
        ("A:9", "A7"),         # 9 degrades to 7
        ("C:6", "C"),          # 6 degrades to major triad
        ("B:dim7", "Bdim"),    # dim7 → dim (triad)
    ])
    def test_valid(self, raw, expected):
        assert _harte_to_shorthand(raw) == expected

    @pytest.mark.parametrize("raw", [
        "N", "", None, "garbage", "Z:min", "H",
    ])
    def test_invalid(self, raw):
        assert _harte_to_shorthand(raw) is None
