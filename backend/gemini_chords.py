"""Cloud chord identification via Gemini + chord-label math.

In hybrid mode Gemini returns the chord set + key for a song and local
Pearson/Viterbi handles the timing. The functions here cover:

- identify_chords_with_gemini: the network call
- _sanitize_chord_label: normalize a free-form chord string into our shorthand
- _infer_better_key: tonic sanity check from the chord timeline
- _diatonic_chords: diatonic 6-chord expansion of a key
"""

from __future__ import annotations

import io
import json
import re
from typing import List, Optional, Tuple

import librosa
import numpy as np

from gemini_client import (
    _gemini_generate_with_retry,
    _get_genai_client,
    _resolve_model,
)
from models import ChordSegment
from templates import _FLAT_TO_SHARP, _KNOWN_QUALITIES, PITCH_NAMES


def _infer_better_key(
    reported_key: str, chord_segments: List[ChordSegment]
) -> Tuple[str, Optional[str]]:
    """Verify or correct the reported key using three independent chord-timeline
    signals — the chord the song starts on, the chord it ends on, and the
    chord it spends the most time on. The actual tonic almost always wins
    on multiple of these.

    Weighted-vote logic:
      first chord    → 2.0 votes for that root
      last chord     → 2.0 votes for that root
      most-played    → 1.5 votes for that root

    If the winning root differs from the reported key's root AND the vote
    margin is strong (≥ 3.5), override with the timeline answer. Mode
    (major/minor) for the new root is decided by whichever quality of that
    root has more time in the timeline.

    Returns (final_key, original_if_overridden_else_None).
    """
    # Drop "N" (no-chord) gaps and very short flicker segments to keep the
    # first/last signals from being noise.
    real = [
        c for c in chord_segments
        if c.label != "N" and (c.end - c.start) > 0.2
    ]
    if len(real) < 3:
        return reported_key, None

    def split_label(label: str) -> Tuple[Optional[str], bool]:
        """Return (root, is_minor_quality). Flats get normalized to sharps."""
        m = re.match(r"^([A-G][#b]?)(.*)$", label)
        if not m:
            return None, False
        root_raw, qual = m.group(1), m.group(2)
        root = _FLAT_TO_SHARP.get(root_raw, root_raw)
        if root not in PITCH_NAMES:
            return None, False
        is_minor = qual == "m" or qual == "m7"
        return root, is_minor

    time_per_root: dict = {}
    major_time_per_root: dict = {}
    minor_time_per_root: dict = {}
    for c in real:
        root, is_minor = split_label(c.label)
        if root is None:
            continue
        dur = c.end - c.start
        time_per_root[root] = time_per_root.get(root, 0.0) + dur
        if is_minor:
            minor_time_per_root[root] = minor_time_per_root.get(root, 0.0) + dur
        else:
            major_time_per_root[root] = major_time_per_root.get(root, 0.0) + dur

    if not time_per_root:
        return reported_key, None

    first_root, _ = split_label(real[0].label)
    last_root, _ = split_label(real[-1].label)
    most_played = max(time_per_root, key=time_per_root.get)

    votes: dict = {}
    if first_root:
        votes[first_root] = votes.get(first_root, 0.0) + 2.0
    if last_root:
        votes[last_root] = votes.get(last_root, 0.0) + 2.0
    votes[most_played] = votes.get(most_played, 0.0) + 1.5

    inferred_root = max(votes, key=votes.get)
    inferred_score = votes[inferred_root]

    parts = (reported_key or "").split()
    if len(parts) != 2:
        return reported_key, None
    reported_root_raw, _reported_mode = parts[0], parts[1].lower()
    reported_root = _FLAT_TO_SHARP.get(reported_root_raw, reported_root_raw)

    if inferred_root == reported_root:
        return reported_key, None

    # Threshold 4.0 requires either first+last agreeing, or all three signals.
    if inferred_score < 4.0:
        return reported_key, None

    minor_time = minor_time_per_root.get(inferred_root, 0.0)
    major_time = major_time_per_root.get(inferred_root, 0.0)
    inferred_mode = "minor" if minor_time > major_time else "major"
    inferred_key = f"{inferred_root} {inferred_mode}"
    # ASCII arrow on purpose — Windows console (cp1252) can't encode '→'.
    print(
        f"[key] overriding {reported_key} -> {inferred_key} "
        f"(first={first_root}, last={last_root}, "
        f"most_played={most_played}, vote_score={inferred_score:.1f})"
    )
    return inferred_key, reported_key


def _diatonic_chords(key: str) -> List[str]:
    """Return the 6 most-common diatonic chords for a major or minor key
    (skipping the seldom-used vii° / ii° as they're rarely the actual chord
    being played in pop / rock).

    Examples:
      "D major" → ["D", "Em", "F#m", "G", "A", "Bm"]
      "E minor" → ["Em", "G", "Am", "Bm", "C", "D"]
    """
    parts = (key or "").strip().split()
    if len(parts) != 2:
        return []
    root_raw, mode = parts[0], parts[1].lower()
    root = _FLAT_TO_SHARP.get(root_raw, root_raw)
    if root not in PITCH_NAMES:
        return []
    root_idx = PITCH_NAMES.index(root)
    if mode == "major":
        intervals = [0, 2, 4, 5, 7, 9]
        qualities = ["", "m", "m", "", "", "m"]
    elif mode == "minor":
        intervals = [0, 3, 5, 7, 8, 10]
        qualities = ["m", "", "m", "m", "", ""]
    else:
        return []
    return [f"{PITCH_NAMES[(root_idx + i) % 12]}{q}" for i, q in zip(intervals, qualities)]


def _sanitize_chord_label(label: str) -> Optional[str]:
    """Normalize a free-form chord label (as returned by Gemini) into our
    internal shorthand. Returns None if the label can't be parsed or uses
    a quality we don't have a template for.
    """
    if not isinstance(label, str):
        return None
    s = label.strip()
    if not s:
        return None

    # Strip slash-bass suffix; the bass note doesn't change the chord template.
    if "/" in s:
        s = s.split("/", 1)[0].strip()

    m = re.match(r"^([A-G])([#b]?)(.*)$", s)
    if not m:
        return None
    root_letter, accidental, qual_raw = m.group(1), m.group(2), m.group(3)
    root = root_letter + accidental

    root = _FLAT_TO_SHARP.get(root, root)
    if root not in PITCH_NAMES:
        return None

    qual = qual_raw.strip()
    if qual.lower() == "maj":
        qual = ""
    elif qual.lower() == "min":
        qual = "m"
    elif qual.lower() == "minor":
        qual = "m"
    elif qual.lower() == "major":
        qual = ""

    if qual not in _KNOWN_QUALITIES:
        return None

    return f"{root}{qual}"


def identify_chords_with_gemini(
    audio: np.ndarray, sr: int, model_override: Optional[str] = None
) -> dict:
    """Send the harmonic audio to Gemini and return:
        {"key": "<str or None>", "chords": ["D", "A", "Em", "G", ...]}

    Raises on any failure (missing key, network, refusal, malformed JSON,
    or sanitization leaving < 2 valid chords) so the caller falls back to
    full-vocabulary local recognition.
    """
    import soundfile as sf
    from google.genai import types as genai_types

    client = _get_genai_client()

    # 16kHz mono keeps the payload tiny and is plenty for chord ID — we're
    # not asking Gemini for fine-grained timing.
    target_sr = 16000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = audio.astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio, target_sr, format="WAV", subtype="PCM_16")
    audio_part = genai_types.Part.from_bytes(
        data=buf.getvalue(), mime_type="audio/wav"
    )

    prompt = (
        "Analyze the chord content of this song.\n"
        "Return JSON only, no commentary, no markdown fences.\n"
        "Schema:\n"
        '  {"key": "<key name like \'D major\' or \'E minor\'>",\n'
        '   "chords": ["<chord1>", "<chord2>", ...]}\n'
        "Rules:\n"
        "- Be COMPREHENSIVE: include EVERY chord you hear in the song, even chords\n"
        "  that only appear briefly (transition chords, bridge chords, one-off\n"
        "  chords). It is much better to include an extra chord than to miss one.\n"
        "- Be precise about major vs minor — never substitute the parallel\n"
        "  (e.g. don't return Am when you hear A major, or D when you hear Dm).\n"
        "- Unique chords only; order doesn't matter.\n"
        "- Use shorthand: root for major (e.g. D), root+m for minor (e.g. Em),\n"
        "  root+7 for dominant 7, root+maj7 / root+m7 for sevenths,\n"
        "  root+sus2 / root+sus4 for suspended, root+dim / root+aug for diminished/augmented.\n"
        "- Use sharps for accidentals (e.g. F#m, not Gbm).\n"
        "- Strip any slash bass (treat D/F# as D).\n"
        '- Key example: "D major". Omit the key field if you cannot determine it.\n'
        "- The tonic is the chord the song most often resolves to — usually\n"
        "  the first chord, the last chord, or the chord that ends each section.\n"
        "- IMPORTANT: a major key and its relative minor share the same chord\n"
        "  set (e.g. D major and E minor both contain D, Em, G, A, Bm, C, F#dim).\n"
        "  Don't default to the minor key just because the chord set fits it —\n"
        "  pick whichever tonic the song actually resolves to.\n"
    )

    response = _gemini_generate_with_retry(
        client,
        model=_resolve_model(model_override),
        contents=[audio_part, prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,  # near-deterministic — same audio should return same chord set
        ),
    )

    raw = (response.text or "").strip()
    if not raw:
        raise ValueError("Gemini chord-ID returned empty response")

    if raw.startswith("```"):
        raw = raw.strip("`")
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1 :]
        raw = raw.strip("`").strip()

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini chord-ID: response is not a JSON object")

    raw_chords = parsed.get("chords")
    if not isinstance(raw_chords, list):
        raise ValueError("Gemini chord-ID: 'chords' missing or not a list")

    sanitized: List[str] = []
    for item in raw_chords:
        clean = _sanitize_chord_label(item if isinstance(item, str) else "")
        if clean and clean not in sanitized:
            sanitized.append(clean)

    if len(sanitized) < 2:
        raise ValueError(
            f"Gemini chord-ID: only {len(sanitized)} valid chord(s) after sanitization"
        )

    key = parsed.get("key")
    if not isinstance(key, str) or not key.strip():
        key = None

    return {"key": key, "chords": sanitized}
