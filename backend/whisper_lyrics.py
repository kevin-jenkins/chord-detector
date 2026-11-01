"""Lyric transcription via faster-whisper (local fallback).

Whisper was trained on speech, not singing — these knobs (peak normalize,
80Hz high-pass, 0.85x time-stretch) make it noticeably more usable on sung
material, but it's still the fallback when the Gemini path is available.
"""

from __future__ import annotations

import gc
import os
from typing import List

import librosa
import numpy as np

from models import LyricWord

_WHISPER_ALLOWED = {"small", "medium", "large-v3"}
_WHISPER_CACHE_MAX = 2

_whisper_models: dict = {}
_whisper_lru: list = []  # most-recently-used at the end


def _get_whisper_model(size: str):
    """Lazy load + LRU-cache faster-whisper models keyed by size name."""
    from faster_whisper import WhisperModel

    if size in _whisper_models:
        _whisper_lru.remove(size)
        _whisper_lru.append(size)
        return _whisper_models[size]

    print(f"Loading Whisper {size} (first run downloads model weights)...")
    model = WhisperModel(size, device="cpu", compute_type="int8")
    _whisper_models[size] = model
    _whisper_lru.append(size)

    # CTranslate2 holds C-side RAM until the Python destructor runs, so drop
    # the local reference AND force a collect — otherwise flipping sizes
    # balloons RSS by 2–3 GB.
    while len(_whisper_lru) > _WHISPER_CACHE_MAX:
        oldest = _whisper_lru.pop(0)
        evicted = _whisper_models.pop(oldest, None)
        del evicted
        gc.collect()

    return model


def transcribe_vocals(
    vocals_audio: np.ndarray,
    sr: int,
    model_size: str = "small",
    initial_prompt: str = "",
) -> List[LyricWord]:
    """Word-level lyric transcription on the isolated vocals stem.

    Returns words sorted by start time. `initial_prompt` biases Whisper's
    vocabulary — useful for band names and unusual proper nouns.
    """
    from scipy.signal import butter, filtfilt

    if model_size not in _WHISPER_ALLOWED:
        model_size = "small"
    env_size = os.environ.get("WHISPER_MODEL")
    if model_size == "small" and env_size in _WHISPER_ALLOWED:
        model_size = env_size

    model = _get_whisper_model(model_size)

    # Peak normalize to ~-1 dBFS so dynamic range into Whisper is consistent.
    peak = float(np.max(np.abs(vocals_audio))) if vocals_audio.size else 0.0
    if peak > 1e-6:
        vocals_audio = (vocals_audio / peak * 0.89).astype(np.float32)

    # 80 Hz high-pass — strips sub-bass rumble that confuses VAD without
    # touching vocal formants (which start above ~250Hz).
    if sr > 200:
        b, a = butter(2, 80.0 / (sr / 2), btype="high")
        vocals_audio = filtfilt(b, a, vocals_audio).astype(np.float32)

    # Slow-audio trick: stretch to 0.85x speed before Whisper sees them.
    # Whisper is empirically more accurate on slower speech; timestamps are
    # scaled back below.
    STRETCH_RATE = 0.85
    if STRETCH_RATE != 1.0:
        vocals_audio = librosa.effects.time_stretch(
            vocals_audio, rate=STRETCH_RATE
        ).astype(np.float32)

    target_sr = 16000
    if sr != target_sr:
        vocals_audio = librosa.resample(
            vocals_audio, orig_sr=sr, target_sr=target_sr
        )
    vocals_audio = vocals_audio.astype(np.float32)

    segments, _info = model.transcribe(
        vocals_audio,
        word_timestamps=True,
        vad_filter=True,
        initial_prompt=initial_prompt or None,
    )

    words: List[LyricWord] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            text = (w.word or "").strip()
            if not text:
                continue
            words.append(
                LyricWord(
                    word=text,
                    start=float(w.start) * STRETCH_RATE,
                    end=float(w.end) * STRETCH_RATE,
                )
            )
    words.sort(key=lambda w: w.start)
    return words
