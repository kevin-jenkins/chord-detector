"""Shared progress-bar state for /analyze and /progress.

Single-user app with a /analyze request lock means we can get away with a
module-global dict; a second concurrent analysis gets rejected at the entry
point before it ever touches this.
"""

from __future__ import annotations

import time as _time
from typing import List

# Per-stage weights (relative time, summed → 100%). analyze() annotates each
# stage boundary; GET /progress reflects the latest pct.
_STAGE_WEIGHTS = {
    "load":           2,   # librosa.load
    "beats":          3,   # beat_track
    "hpss":          10,   # only when separate_stems = False
    "stems":         30,   # demucs with shifts=0 (was 60 with shifts=1)
    "gemini":        25,   # parallel chord_id + lyric Gemini calls
    "tuning_key":     2,   # estimate_tuning + estimate_key
    "chords":         5,   # extract_chords (Pearson + Viterbi)
    "sections":       1,   # detect_sections (chroma boundary + KMeans)
    "lyrics_whisper": 25,  # Whisper transcribe (primary or fallback)
}

_progress = {"stage": "idle", "pct": 0}
_progress_total = 1
_progress_done = 0
_stage_started_at: dict = {}


def _progress_begin(stages: List[str]) -> None:
    global _progress_total, _progress_done
    _progress_total = max(1, sum(_STAGE_WEIGHTS[s] for s in stages))
    _progress_done = 0
    _progress["stage"] = "starting"
    _progress["pct"] = 0


def _progress_enter(stage: str) -> None:
    _progress["stage"] = stage
    _progress["pct"] = int(_progress_done / _progress_total * 100)
    _stage_started_at[stage] = _time.time()


def _progress_complete(stage: str) -> None:
    global _progress_done
    _progress_done += _STAGE_WEIGHTS[stage]
    _progress["pct"] = int(_progress_done / _progress_total * 100)
    started = _stage_started_at.pop(stage, None)
    if started is not None:
        print(f"[stage] {stage}: {_time.time() - started:.1f}s")


def _progress_extend(stage_names: List[str]) -> None:
    """Dynamically add stages to the running total — used when a stage's
    necessity is only known after earlier stages finish (e.g., the Whisper
    fallback only fires when Gemini lyric transcription fails)."""
    global _progress_total
    additional = sum(_STAGE_WEIGHTS[s] for s in stage_names)
    if additional <= 0:
        return
    _progress_total += additional
    _progress["pct"] = int(_progress_done / _progress_total * 100)
