"""Cloud lyric transcription via Gemini (hybrid mode default).

Multimodal — meaningfully better than Whisper on sung material because the
training data is broader. Raises on any failure so the caller can fall back
to Whisper transparently.
"""

from __future__ import annotations

import io
import json
from typing import List, Optional

import numpy as np

from gemini_client import (
    _gemini_generate_with_retry,
    _get_genai_client,
    _resolve_model,
)
from models import LyricWord


def transcribe_with_gemini(
    vocals_audio: np.ndarray,
    sr: int,
    initial_prompt: str = "",
    model_override: Optional[str] = None,
) -> List[LyricWord]:
    """Return a time-aligned list of sung words. Raises on any failure."""
    import soundfile as sf
    from google.genai import types as genai_types

    client = _get_genai_client()

    # Resample to 16kHz mono before encoding. Gemini's inline-audio limit
    # is ~20MB; a 7-min song at the demucs native rate (44.1kHz, mono, 16-bit)
    # is ~37MB, well over the limit. 16kHz is plenty for word recognition
    # and matches what the chord-ID path does.
    import librosa
    target_sr = 16000
    if sr != target_sr:
        vocals_audio = librosa.resample(
            vocals_audio, orig_sr=sr, target_sr=target_sr
        )
    import numpy as np
    vocals_audio = vocals_audio.astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, vocals_audio, target_sr, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()
    audio_part = genai_types.Part.from_bytes(
        data=audio_bytes, mime_type="audio/wav"
    )

    hint_line = (
        f"Context (may help with proper nouns, genre, language): {initial_prompt.strip()}"
        if initial_prompt and initial_prompt.strip()
        else "Context: none provided"
    )

    # Deliberately neutral prompt — no examples of lyrics, no expected words.
    prompt = (
        "Transcribe the singing in this audio.\n"
        "Return JSON only, no commentary, no markdown fences.\n"
        "Schema:\n"
        '  {"words": [{"word": "<single sung word>", '
        '"start": <seconds>, "end": <seconds>}, ...]}\n'
        "Rules:\n"
        "- One JSON object per word as actually sung.\n"
        "- start and end are seconds from the beginning of the audio.\n"
        "- Skip purely instrumental sections (no entries).\n"
        "- If a word is ambiguous, prefer omission over guessing.\n"
        "- No punctuation; preserve capitalization only for 'I' and proper nouns.\n"
        f"{hint_line}\n"
    )

    response = _gemini_generate_with_retry(
        client,
        model=_resolve_model(model_override),
        contents=[audio_part, prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,  # low variance — not strictly deterministic, but close
        ),
    )

    raw = (response.text or "").strip()
    if not raw:
        raise ValueError("Gemini returned empty response")

    # Defensive: strip code fences if the model wrapped its JSON anyway.
    if raw.startswith("```"):
        raw = raw.strip("`")
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1 :]
        raw = raw.strip("`").strip()

    def _diag(reason: str) -> ValueError:
        snippet = raw[:500].replace("\n", " ")
        return ValueError(
            f"{reason}; response snippet (first 500 chars): {snippet!r}"
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _diag(f"JSON decode failed: {e}") from None

    # Lenient shape detection. Gemini sometimes returns the list under a
    # different key, or as a bare top-level array. Accept what looks right.
    items = None
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        for key in ("words", "transcript", "lyrics", "results", "data"):
            v = parsed.get(key)
            if isinstance(v, list):
                items = v
                break
            if isinstance(v, dict):
                for inner in ("words", "transcript", "lyrics", "results"):
                    w = v.get(inner)
                    if isinstance(w, list):
                        items = w
                        break
                if items is not None:
                    break
    if items is None:
        raise _diag(
            f"could not locate a list of words in response "
            f"(keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'top-level not a dict'})"
        )

    # Lenient per-word shape: accept several field-name variants.
    _WORD_KEYS = ("word", "text", "token", "content", "value")
    _START_KEYS = ("start", "start_time", "startTime", "begin", "t0", "from")
    _END_KEYS = ("end", "end_time", "endTime", "stop", "t1", "to")

    def _pick(d: dict, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    words: List[LyricWord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text_val = _pick(item, _WORD_KEYS)
        text = str(text_val).strip() if text_val is not None else ""
        if not text:
            continue
        s_val = _pick(item, _START_KEYS)
        e_val = _pick(item, _END_KEYS)
        try:
            start = float(s_val) if s_val is not None else 0.0
            end = float(e_val) if e_val is not None else start
        except (TypeError, ValueError):
            continue
        words.append(LyricWord(word=text, start=start, end=end))

    words.sort(key=lambda w: w.start)
    return words
