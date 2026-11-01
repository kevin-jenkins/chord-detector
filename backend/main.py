"""Chord Detector backend — FastAPI app + routes + /analyze orchestration.

The pipeline is split across sibling modules:

  templates.py        chord template library + music-theory constants
  models.py           Pydantic response models
  cache.py            result cache (disk+memory) + stems sub-cache
  progress.py         shared progress-bar state
  gemini_client.py    shared Gen AI client + retry/backoff
  gemini_lyrics.py    cloud lyric transcription
  gemini_chords.py    cloud chord identification + chord-label math
  stems.py            demucs stem separation
  whisper_lyrics.py   faster-whisper local lyric transcription
  chords.py           chroma_cqt → Pearson → Viterbi local chord recognition
  sections.py         song-structure clustering
"""

from __future__ import annotations

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

# Load backend/.env (if present) so API keys etc. can be set without touching
# the user's shell env. Must happen BEFORE any os.environ reads below.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv is optional; falls back to OS env only

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from cache import _audio_hash, _cache_get, _cache_key, _cache_put
from chords import estimate_key, extract_chords
from gemini_chords import (
    _diatonic_chords,
    _infer_better_key,
    identify_chords_with_gemini,
)
from gemini_client import _GEMINI_MODELS, _log_gemini_failure
from gemini_lyrics import transcribe_with_gemini
from models import AnalysisResult, LyricWord
from progress import (
    _progress,
    _progress_begin,
    _progress_complete,
    _progress_enter,
    _progress_extend,
)
from sections import detect_sections
from stems import run_stem_separation
from templates import PITCH_NAMES
from whisper_lyrics import _WHISPER_ALLOWED, transcribe_vocals

_LYRIC_PROVIDERS = {"gemini", "whisper"}
_CHORD_PROVIDERS = {"hybrid", "local"}

# Request-level serialization: single-user local app, so if a second /analyze
# arrives mid-analysis we return 409 rather than letting the two trample
# shared progress + cache state.
_analyze_lock = threading.Lock()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Chord Detector", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/progress")
def progress():
    return _progress


@app.post("/analyze", response_model=AnalysisResult)
def analyze(
    file: UploadFile = File(...),
    separate_stems: bool = Form(False),
    vocabulary: str = Form("triads"),
    transcribe_lyrics: bool = Form(False),
    whisper_model: str = Form("small"),
    lyric_hint: str = Form(""),
    lyric_provider: str = Form("gemini"),
    chord_provider: str = Form("hybrid"),
    gemini_model: str = Form(""),
):
    if lyric_provider not in _LYRIC_PROVIDERS:
        lyric_provider = "gemini"
    if chord_provider not in _CHORD_PROVIDERS:
        chord_provider = "hybrid"
    # Per-request override (form value > env default). Empty / invalid →
    # use the env-configured _GEMINI_MODEL.
    gemini_model_override = gemini_model if gemini_model in _GEMINI_MODELS else None
    # NOTE: sync handler on purpose. The heavy work (librosa, demucs, whisper)
    # is all blocking, so running this on the asyncio event loop would freeze
    # GET /progress until analysis finishes. FastAPI runs sync handlers in its
    # threadpool, which keeps the event loop free for the poller.

    # Single-user app with shared progress state: refuse a second concurrent
    # analysis rather than let it corrupt the first's state and progress bar.
    if not _analyze_lock.acquire(blocking=False):
        raise HTTPException(
            409, "Another analysis is already running — wait for it to finish"
        )

    if not file.filename:
        _analyze_lock.release()
        raise HTTPException(400, "No filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}:
        _analyze_lock.release()
        raise HTTPException(400, f"Unsupported format: {suffix}")

    if transcribe_lyrics and not separate_stems:
        # Whisper on a full mix produces garbage; the isolated vocals stem
        # is what makes transcription tolerable on songs with backing music.
        _progress["stage"] = "idle"
        _progress["pct"] = 0
        _analyze_lock.release()
        raise HTTPException(
            400, "transcribe_lyrics requires separate_stems=true"
        )

    # Read the uploaded bytes once so we can compute a cache key before
    # spending any compute. Sync read because we're a sync handler.
    body_bytes = file.file.read()

    # Cache lookup. Same audio + same params → instant return.
    ck = _cache_key(
        body_bytes,
        separate_stems=separate_stems,
        vocabulary=vocabulary,
        transcribe_lyrics=transcribe_lyrics,
        whisper_model=whisper_model,
        lyric_hint=lyric_hint,
        lyric_provider=lyric_provider,
        chord_provider=chord_provider,
        gemini_model=gemini_model,
    )
    cached = _cache_get(ck)
    if cached is not None:
        _progress["stage"] = "done"
        _progress["pct"] = 100
        print(f"[cache] hit ({len(body_bytes)} bytes) — returning cached result")
        _analyze_lock.release()
        return AnalysisResult(**cached)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(body_bytes)
        tmp_path = tmp.name

    # Declare which stages will run so the progress bar weights are right.
    # The "gemini" stage covers the parallel chord_id + lyric calls.
    # "lyrics_whisper" is added up front only when Whisper is the explicit
    # provider; the fallback path (Gemini failed → Whisper) adds it
    # dynamically via _progress_extend.
    want_chord_gemini = chord_provider == "hybrid"
    want_lyric_gemini = transcribe_lyrics and lyric_provider == "gemini"
    stages = ["load", "beats"]
    stages.append("stems" if separate_stems else "hpss")
    if want_chord_gemini or want_lyric_gemini:
        stages.append("gemini")
    stages += ["tuning_key", "chords", "sections"]
    if transcribe_lyrics and not want_lyric_gemini:
        stages.append("lyrics_whisper")
    _progress_begin(stages)

    try:
        _progress_enter("load")
        # Always load the full mix at 22050 for beat tracking — beats are
        # easier to find with drums in the signal.
        y_full, sr = librosa.load(tmp_path, sr=22050, mono=True)
        duration = float(librosa.get_duration(y=y_full, sr=sr))
        _progress_complete("load")

        _progress_enter("beats")
        tempo, beat_frames = librosa.beat.beat_track(y=y_full, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        bpm = float(np.atleast_1d(tempo)[0])
        _progress_complete("beats")

        vocals_audio: Optional[np.ndarray] = None
        vocals_sr: Optional[int] = None

        if separate_stems:
            _progress_enter("stems")
            # Demucs: drop drums, keep bass + other for chord recognition.
            # If lyrics are requested, also pull out the isolated vocals stem.
            y_sep, vocals_audio, sep_sr = run_stem_separation(
                tmp_path, want_vocals=transcribe_lyrics,
                audio_hash=_audio_hash(body_bytes),
            )
            if sep_sr != sr:
                y_sep = librosa.resample(y_sep, orig_sr=sep_sr, target_sr=sr)
            y_harm = y_sep
            vocals_sr = sep_sr
            _progress_complete("stems")
        else:
            _progress_enter("hpss")
            # Fallback: HPSS as a poor-man's stem separation. Removes
            # transient/percussive content but leaves vocals + harmonics intact.
            y_harm, _ = librosa.effects.hpss(y_full, margin=3.0)
            _progress_complete("hpss")

        # Parallel Gemini block — chord_id and lyric transcription are both
        # independent network-bound calls; running them concurrently saves
        # the wall time of the shorter of the two.
        gemini_chord_set: Optional[List[str]] = None
        gemini_key: Optional[str] = None
        chord_provider_used: str = "local"
        lyrics: Optional[List[LyricWord]] = None
        lyric_provider_used: Optional[str] = None
        run_lyric_gemini = (
            want_lyric_gemini
            and vocals_audio is not None
            and vocals_sr is not None
        )

        if want_chord_gemini or run_lyric_gemini:
            _progress_enter("gemini")
            with ThreadPoolExecutor(max_workers=2) as ex:
                chord_fut = (
                    ex.submit(
                        identify_chords_with_gemini,
                        y_harm, sr, gemini_model_override,
                    )
                    if want_chord_gemini
                    else None
                )
                lyric_fut = (
                    ex.submit(
                        transcribe_with_gemini,
                        vocals_audio, vocals_sr,
                        lyric_hint, gemini_model_override,
                    )
                    if run_lyric_gemini
                    else None
                )

                if chord_fut is not None:
                    try:
                        info = chord_fut.result()
                        gemini_chord_set = info["chords"]
                        gemini_key = info.get("key")
                        chord_provider_used = "hybrid"
                    except Exception as e:
                        print(
                            f"Gemini chord identification failed "
                            f"({type(e).__name__}: {e}); "
                            f"falling back to full local template set"
                        )
                        _log_gemini_failure("chord-id", e)
                        chord_provider_used = "local-fallback"

                if lyric_fut is not None:
                    try:
                        lyrics = lyric_fut.result()
                        lyric_provider_used = "gemini"
                    except Exception as e:
                        print(
                            f"Gemini transcription failed "
                            f"({type(e).__name__}: {e}); "
                            f"falling back to Whisper small"
                        )
                        _log_gemini_failure("lyrics", e)
                        # lyrics stays None — Whisper fallback fires below
            _progress_complete("gemini")

        _progress_enter("tuning_key")
        # Tuning offset (in fractions of a semitone). Without this, chroma bins
        # smear into neighbours when the band is even slightly flat.
        tuning = float(librosa.estimate_tuning(y=y_harm, sr=sr))
        ks_key = estimate_key(y_harm, sr, tuning)
        _progress_complete("tuning_key")

        # Gemini wins on key when it returned one; otherwise KS.
        key = gemini_key or ks_key

        # Augment Gemini's chord set with two safety nets:
        #   (1) Parallel major/minor of every Gemini chord (D ↔ Dm).
        #   (2) Diatonic chords of Gemini's key, when it returned one.
        # The UI still shows the original gemini_chord_set so the user sees
        # exactly what Gemini reported.
        allowed_for_matcher: Optional[List[str]] = None
        if gemini_chord_set:
            allowed_for_matcher = list(gemini_chord_set)
            for c in list(allowed_for_matcher):
                if c in PITCH_NAMES and f"{c}m" not in allowed_for_matcher:
                    allowed_for_matcher.append(f"{c}m")
                elif c.endswith("m") and not c.endswith(("dim", "aug")):
                    root = c[:-1]
                    if root in PITCH_NAMES and root not in allowed_for_matcher:
                        allowed_for_matcher.append(root)
            if gemini_key:
                for c in _diatonic_chords(gemini_key):
                    if c not in allowed_for_matcher:
                        allowed_for_matcher.append(c)

        _progress_enter("chords")
        try:
            chords = extract_chords(
                y_harm, sr, duration, tuning, vocabulary,
                allowed_chord_labels=allowed_for_matcher,
            )
        except ValueError as e:
            # The Gemini-restricted template set was too thin / all out-of-
            # vocab. Drop the restriction and run with the full vocabulary.
            print(
                f"Restricted chord matching failed ({e}); "
                f"retrying with full vocabulary"
            )
            chords = extract_chords(y_harm, sr, duration, tuning, vocabulary)
            if chord_provider_used == "hybrid":
                chord_provider_used = "local-fallback"
            gemini_chord_set = None
        _progress_complete("chords")

        # Tonic sanity check from chord-timeline signals.
        key, _key_was = _infer_better_key(key, chords)

        _progress_enter("sections")
        sections = detect_sections(y_harm, sr, duration, beat_frames)
        _progress_complete("sections")

        # Whisper path: fires either as the primary provider or as the
        # fallback when Gemini lyric transcription failed above.
        if (
            transcribe_lyrics
            and vocals_audio is not None
            and vocals_sr is not None
            and lyrics is None
        ):
            if want_lyric_gemini:
                _progress_extend(["lyrics_whisper"])
            _progress_enter("lyrics_whisper")
            lyrics = transcribe_vocals(
                vocals_audio,
                vocals_sr,
                model_size="small" if want_lyric_gemini else whisper_model,
                initial_prompt=lyric_hint,
            )
            lyric_provider_used = (
                "whisper-fallback" if want_lyric_gemini else "whisper"
            )
            _progress_complete("lyrics_whisper")

        result = AnalysisResult(
            duration=duration,
            bpm=bpm,
            key=key,
            beats=beat_times,
            chords=chords,
            lyrics=lyrics,
            sections=sections if sections else None,
            lyric_provider_used=lyric_provider_used,
            chord_provider_used=chord_provider_used,
            gemini_chord_set=gemini_chord_set,
            gemini_key=gemini_key,
        )
        # Stash for next time. Skip when Gemini fell back — caching a fallback
        # would short-circuit every future retry even after Google's outage
        # clears.
        fell_back = (
            chord_provider_used == "local-fallback"
            or lyric_provider_used == "whisper-fallback"
        )
        if fell_back:
            print(f"[cache] skip put (fallback path: chord={chord_provider_used}, lyric={lyric_provider_used})")
        else:
            _cache_put(ck, result.model_dump())
        return result
    finally:
        # Settle the bar at 100% so the frontend's final poll sees completion
        # before it stops polling.
        _progress["stage"] = "done"
        _progress["pct"] = 100
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        _analyze_lock.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
