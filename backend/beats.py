"""Beat tracking — pluggable backend.

librosa.beat.beat_track is the universal fallback. Set BEAT_TRACKER=beat_this
to use CPJKU's "Beat This!" transformer tracker (ISMIR 2024) instead — more
accurate on syncopated material and produces downbeats, which librosa does
poorly. Beat This! needs its own install:

    pip install beat-this

If import fails at runtime we silently fall back to librosa so a missing
install never breaks a request. See:
    https://pypi.org/project/beat-this/
"""

from __future__ import annotations

import os
from typing import Tuple

import librosa
import numpy as np

_TRACKER = os.environ.get("BEAT_TRACKER", "librosa").lower()


def beat_track(
    y: np.ndarray, sr: int, audio_path: str
) -> Tuple[float, np.ndarray]:
    """Return (bpm, beat_frames) in librosa's shape.

    `audio_path` is the on-disk tmp file librosa.load produced `y` from —
    Beat This! takes a file path and does its own decode, so we pass it
    along rather than re-encoding `y`.
    """
    if _TRACKER == "beat_this":
        try:
            return _beat_this_track(audio_path, sr)
        except Exception as e:
            print(
                f"[beats] Beat This! failed ({type(e).__name__}: {e}); "
                f"falling back to librosa"
            )

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    return bpm, beat_frames


def _beat_this_track(
    audio_path: str, sr: int
) -> Tuple[float, np.ndarray]:
    """Beat This! inference — returns beat times, converted to frames at `sr`.

    Uses librosa's default hop (512) so downstream frame math is consistent.
    """
    from beat_this.inference import File2Beats

    # 'final0' is the shipping checkpoint; CPU is fine for ~1s inference.
    tracker = File2Beats(checkpoint_path="final0", device="cpu")
    beat_times, _downbeat_times = tracker(audio_path)

    beat_times = np.asarray(beat_times, dtype=float)
    if beat_times.size < 2:
        # Too few beats to compute tempo — bail and let caller fall back.
        raise ValueError(f"Beat This! returned only {beat_times.size} beat(s)")

    # Median inter-beat interval → BPM. More robust than mean on songs with
    # a short count-in or a trailing ritardando.
    intervals = np.diff(beat_times)
    bpm = float(60.0 / np.median(intervals))

    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    return bpm, beat_frames
