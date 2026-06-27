"""
Chord Detector backend.

POST /analyze  -> multipart upload of an audio file
                  returns { key, bpm, duration, chords: [...], beats: [...] }

Uses HPSS + librosa chroma_cqt + cosine-similarity template matching for
chord recognition, and librosa beat_track + Krumhansl-Schmuckler for BPM/key.
Audio never leaves the machine.
"""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

# Load backend/.env (if present) so API keys etc. can be set without touching
# the user's shell env. Must happen BEFORE any os.environ reads below.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv is optional; falls back to OS env only

import librosa
import numpy as np
from scipy.ndimage import median_filter
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Heavy ML deps are lazily imported inside their respective functions so the
# rest of the app stays snappy at startup.
_demucs_model = None    # cached across requests
_demucs_device = None   # resolved on first stem-separation request

# Whisper supports per-request model size selection ("small" / "medium" / "large-v3").
# We keep an LRU-capped cache so users can toggle between sizes without paying
# the load cost every time. Keys are size names; values are WhisperModel objects.
_whisper_models: dict = {}
_whisper_lru: list = []          # most-recently-used at the end
_WHISPER_CACHE_MAX = 2           # keep at most this many models warm
_WHISPER_ALLOWED = {"small", "medium", "large-v3"}

# Gemini transcription (cloud). Multimodal model — significantly better than
# Whisper on out-of-distribution audio like singing. Free tier via AI Studio.
_genai_client = None
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
_LYRIC_PROVIDERS = {"gemini", "whisper"}
_CHORD_PROVIDERS = {"hybrid", "local"}
_GEMINI_MODELS = {"gemini-3.5-flash", "gemini-2.5-pro"}

# Per-stage weights (relative time, summed → 100%). analyze() annotates each
# stage boundary; GET /progress reflects the latest pct. Single-user local-only
# constraint means we don't need per-request isolation.
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

# Result cache keyed by sha256(audio_bytes + sorted params). Two-tier:
#   in-memory LRU (fast hot path) + on-disk JSON (survives backend restart).
# Single-user local app, so no concurrency concerns.
_RESULT_CACHE: dict = {}
_RESULT_CACHE_MAX = 32
_RESULT_CACHE_DIR = Path(__file__).parent / ".cache" / "results"
_RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Stems sub-cache keyed by sha256(audio_bytes). Stems depend only on the
# audio itself, so flipping any param (Pro/Flash, vocabulary, hint, …)
# doesn't invalidate them. In-memory only — stems are big (~170MB each)
# and the use case is "same session, knob-fiddling iteration".
_STEMS_CACHE: dict = {}
_STEMS_CACHE_MAX = 4


def _cache_key(audio_bytes: bytes, **params) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(audio_bytes)
    param_str = "|".join(f"{k}={params[k]}" for k in sorted(params))
    h.update(param_str.encode("utf-8"))
    return h.hexdigest()


def _audio_hash(audio_bytes: bytes) -> str:
    import hashlib
    return hashlib.sha256(audio_bytes).hexdigest()


def _cache_get(key: str):
    # Memory first
    val = _RESULT_CACHE.pop(key, None)
    if val is not None:
        _RESULT_CACHE[key] = val  # touch LRU
        return val
    # Disk fall-through
    disk_path = _RESULT_CACHE_DIR / f"{key}.json"
    if disk_path.exists():
        try:
            import json as _json
            with open(disk_path, "r", encoding="utf-8") as f:
                val = _json.load(f)
            _RESULT_CACHE[key] = val  # hydrate memory
            print(f"[cache] disk hit for {key[:12]}...")
            return val
        except Exception as e:
            print(f"[cache] disk read failed for {key[:12]}...: {e}")
    return None


def _cache_put(key: str, value) -> None:
    # Memory tier
    _RESULT_CACHE[key] = value
    while len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
        oldest = next(iter(_RESULT_CACHE))
        _RESULT_CACHE.pop(oldest, None)
    # Disk tier — best-effort; never let a write failure break the request
    try:
        import json as _json
        tmp = _RESULT_CACHE_DIR / f".{key}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(value, f)
        tmp.replace(_RESULT_CACHE_DIR / f"{key}.json")
    except Exception as e:
        print(f"[cache] disk write failed for {key[:12]}...: {e}")


def _stems_cache_get(key: str, need_vocals: bool):
    entry = _STEMS_CACHE.pop(key, None)
    if entry is None:
        return None
    _STEMS_CACHE[key] = entry  # touch LRU regardless
    _, vocals, _ = entry
    if need_vocals and vocals is None:
        # Cached entry doesn't include vocals; force a recompute.
        return None
    return entry


def _stems_cache_put(key: str, harm, vocals, sr) -> None:
    _STEMS_CACHE[key] = (harm, vocals, sr)
    while len(_STEMS_CACHE) > _STEMS_CACHE_MAX:
        oldest = next(iter(_STEMS_CACHE))
        _STEMS_CACHE.pop(oldest, None)


def _progress_begin(stages: List[str]) -> None:
    global _progress_total, _progress_done
    _progress_total = max(1, sum(_STAGE_WEIGHTS[s] for s in stages))
    _progress_done = 0
    _progress["stage"] = "starting"
    _progress["pct"] = 0


def _progress_enter(stage: str) -> None:
    _progress["stage"] = stage
    _progress["pct"] = int(_progress_done / _progress_total * 100)
    import time as _time
    _stage_started_at[stage] = _time.time()


_stage_started_at: dict = {}


def _progress_complete(stage: str) -> None:
    global _progress_done
    _progress_done += _STAGE_WEIGHTS[stage]
    _progress["pct"] = int(_progress_done / _progress_total * 100)
    # Log wall-clock duration for diagnostic visibility.
    import time as _time
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
    # Renormalize pct: the same amount of work is "done" but the total grew.
    _progress["pct"] = int(_progress_done / _progress_total * 100)

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
# Chord template library
# ---------------------------------------------------------------------------

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Three vocabularies, in order of restrictiveness. Smaller vocabularies are
# more stable on noisy material (fewer confusable templates) but can only
# label chords approximately if the actual chord is a 7th/sus/etc.
_VOCAB_SETS = {
    "triads": {
        "":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "m":    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    },
    "triads_7ths": {
        "":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "m":    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "7":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "m7":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    },
    "extended": {
        "":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "m":    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "7":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "m7":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        "sus2": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "sus4": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        "dim":  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "aug":  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    },
}

def _center_and_normalize(matrix: np.ndarray, axis: int) -> np.ndarray:
    """Subtract mean along axis, then unit-normalize along the same axis.
    Result is suitable for Pearson correlation via matrix multiplication."""
    centered = matrix - matrix.mean(axis=axis, keepdims=True)
    norms = np.linalg.norm(centered, axis=axis, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return centered / norms


def _build_template_set(base_templates: dict) -> Tuple[List[str], np.ndarray]:
    """Returns (labels, pearson-ready template matrix)."""
    labels: List[str] = []
    rows = []
    for quality, base in base_templates.items():
        b = np.array(base, dtype=float)
        b /= np.linalg.norm(b)
        for root in range(12):
            rows.append(np.roll(b, root))
            labels.append(f"{PITCH_NAMES[root]}{quality}")
    matrix = np.array(rows)
    return labels, _center_and_normalize(matrix, axis=1)


# Pre-built template sets keyed by vocabulary name → (labels, pearson matrix)
_TEMPLATES: dict = {
    name: _build_template_set(base) for name, base in _VOCAB_SETS.items()
}

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles
# ---------------------------------------------------------------------------

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ChordSegment(BaseModel):
    start: float
    end: float
    label: str


class LyricWord(BaseModel):
    word: str
    start: float
    end: float


class Section(BaseModel):
    start: float
    end: float
    label: str   # e.g. "Intro", "Verse 1", "Chorus", "Bridge", "Outro"


class AnalysisResult(BaseModel):
    duration: float
    bpm: float
    key: str
    beats: List[float]
    chords: List[ChordSegment]
    lyrics: Optional[List[LyricWord]] = None
    sections: Optional[List[Section]] = None
    # Renamed from `provider_used` for clarity now that chord detection has
    # its own provider too.
    lyric_provider_used: Optional[str] = None   # "gemini" | "whisper" | "whisper-fallback"
    chord_provider_used: Optional[str] = None   # "hybrid" | "local" | "local-fallback"
    gemini_chord_set: Optional[List[str]] = None
    gemini_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Stem separation (demucs)
# ---------------------------------------------------------------------------


def _get_demucs_device():
    """Pick the best inference device for demucs once per process.

    Order of preference:
      1. CUDA — NVIDIA discrete GPUs on Linux/Windows. Native complex-tensor
         support; STFT-based audio models work out of the box.
      2. MPS — Apple Silicon (M1/M2/M3). PyTorch 2.x adds complex-tensor
         support; we also flip on PYTORCH_ENABLE_MPS_FALLBACK so any ops the
         Metal backend doesn't implement transparently fall back to CPU.
      3. CPU — universal fallback. The CPU path was DirectML's grave
         (Intel iGPU rejects complex floats), so we don't try DirectML.

    Result is cached for the rest of the process lifetime so subsequent
    requests skip the probe.
    """
    global _demucs_device
    if _demucs_device is not None:
        return _demucs_device
    try:
        import torch
        if torch.cuda.is_available():
            _demucs_device = "cuda"
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "unknown"
            print(f"Demucs device: CUDA ({name})")
            return _demucs_device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            _demucs_device = "mps"
            print("Demucs device: MPS (Apple Silicon, CPU fallback enabled)")
            return _demucs_device
    except Exception as e:
        print(f"GPU detection failed ({type(e).__name__}: {e}); using CPU")
    _demucs_device = "cpu"
    print("Demucs device: CPU")
    return _demucs_device


def run_stem_separation(
    audio_path: str, want_vocals: bool = False,
    audio_hash: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """
    Run demucs and return:
      - harmonic mono mix (bass + other) at the model's native sample rate
      - vocals mono at the model's native sample rate, or None if not requested
      - sample rate

    Drops drums for chord recognition (noise) and discards them entirely.
    Keeps bass so the chord root survives. Vocals are isolated separately so
    Whisper has a clean signal to transcribe — sending it the full mix or
    even the harmonic mix produces garbage.

    First call downloads ~80MB of model weights to ~/.cache/torch/hub.
    Model is cached at module level so subsequent requests are fast.

    Stems sub-cache: if `audio_hash` matches a previous run on the same
    audio bytes, return the cached stems directly and skip demucs entirely.
    Lets the user toggle Gemini Pro/Flash, vocabulary, hint, etc. on the
    same upload without re-running the 165s demucs pass.
    """
    # Stems cache lookup — only requires the audio hash. Stems are
    # determined by the audio alone; no other request param affects them.
    if audio_hash is not None:
        hit = _stems_cache_get(audio_hash, need_vocals=want_vocals)
        if hit is not None:
            harm_cached, vocals_cached, sr_cached = hit
            print(f"[stems-cache] hit for {audio_hash[:12]}...")
            return (
                harm_cached,
                vocals_cached if want_vocals else None,
                sr_cached,
            )

    global _demucs_model
    import sys
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    device = _get_demucs_device()

    # Windows-specific: PyTorch's CPU thread auto-detection under-counts
    # E-cores on 12th-gen+ Intel hybrid CPUs. Force the full physical count.
    # No-op on macOS/Linux where the default works correctly, and a no-op
    # when we're not running on CPU.
    if sys.platform == "win32" and device == "cpu":
        try:
            cpu_threads = max(1, os.cpu_count() or 1)
            torch.set_num_threads(cpu_threads)
        except Exception:
            pass

    if _demucs_model is None:
        # mdx_extra_q: quantized MDX-Net model. ~2x faster than htdemucs_ft
        # on CPU with a modest quality drop in vocal isolation. Acceptable
        # since Gemini does the heavy lifting on chord identification —
        # the chord matcher and Gemini are both robust to mild stem noise.
        m = get_model("mdx_extra_q")
        m.to(device).eval()
        _demucs_model = m

    model = _demucs_model
    sr = int(model.samplerate)

    wav = AudioFile(audio_path).read(
        streams=0, samplerate=sr, channels=model.audio_channels
    )

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    wav_norm = (wav - mean) / (std + 1e-8)

    with torch.no_grad():
        def _run(dev):
            return apply_model(
                model, wav_norm[None],
                device=dev, progress=False,
                shifts=0,      # disable default extra random-time-shift pass
                overlap=0.1,   # default 0.25 → 0.1 trims redundant chunk compute
            )[0]

        try:
            sources = _run(device)
        except RuntimeError as e:
            # Graceful fallback if a GPU device hits an unsupported op
            # mid-run (MPS especially can still surprise on STFT-heavy
            # models). Permanent downgrade for the rest of this process
            # — no per-request flapping.
            if device == "cpu":
                raise
            global _demucs_device
            print(
                f"{device} inference failed ({type(e).__name__}: {e}); "
                f"falling back to CPU for the rest of the process"
            )
            _demucs_device = "cpu"
            model.to("cpu")
            sources = _run("cpu")
    sources = sources * std + mean

    src_names = list(model.sources)
    bass = sources[src_names.index("bass")]
    other = sources[src_names.index("other")]
    harm_mono = (bass + other).mean(dim=0).cpu().numpy().astype(np.float32)

    # Always extract vocals too if we have a cache key — costs ~10ms but
    # means future requests on the same audio that DO want lyrics will hit
    # the stems cache cleanly. Without a key we'd just be wasting memory.
    vocals_mono: Optional[np.ndarray] = None
    if want_vocals or audio_hash is not None:
        vocals = sources[src_names.index("vocals")]
        vocals_mono = vocals.mean(dim=0).cpu().numpy().astype(np.float32)

    if audio_hash is not None:
        _stems_cache_put(audio_hash, harm_mono, vocals_mono, sr)

    return harm_mono, vocals_mono if want_vocals else None, sr


# ---------------------------------------------------------------------------
# Lyric transcription (faster-whisper)
# ---------------------------------------------------------------------------


def _get_whisper_model(size: str):
    """Lazy load + LRU-cache faster-whisper models keyed by size name."""
    from faster_whisper import WhisperModel

    if size in _whisper_models:
        # Touch LRU
        _whisper_lru.remove(size)
        _whisper_lru.append(size)
        return _whisper_models[size]

    print(f"Loading Whisper {size} (first run downloads model weights)...")
    model = WhisperModel(size, device="cpu", compute_type="int8")
    _whisper_models[size] = model
    _whisper_lru.append(size)

    # LRU evict if over cap. Cheap since model objects hold ~500MB+; releasing
    # the reference lets Python free it on the next GC.
    while len(_whisper_lru) > _WHISPER_CACHE_MAX:
        oldest = _whisper_lru.pop(0)
        _whisper_models.pop(oldest, None)

    return model


def transcribe_vocals(
    vocals_audio: np.ndarray,
    sr: int,
    model_size: str = "small",
    initial_prompt: str = "",
) -> List[LyricWord]:
    """
    Word-level lyric transcription via faster-whisper on the isolated vocals
    stem. Returns words sorted by start time.

    Pre-processes the vocals stem (peak normalize + 80Hz high-pass) so quiet
    or rumbly stems behave uniformly. Decoding uses a wider beam, disables
    cross-segment text conditioning (cuts hallucinations on instrumental gaps),
    and a temperature ladder (only escalates when low-confidence).

    `initial_prompt` is passed directly to Whisper as a vocabulary bias —
    useful for band names and unusual proper nouns.
    """
    from scipy.signal import butter, filtfilt

    if model_size not in _WHISPER_ALLOWED:
        model_size = "small"
    # Allow env override to win (existing behavior); explicit per-request
    # selection still beats the env when set to a valid value.
    env_size = os.environ.get("WHISPER_MODEL")
    if model_size == "small" and env_size in _WHISPER_ALLOWED:
        model_size = env_size

    model = _get_whisper_model(model_size)

    # ---- Preprocessing ----------------------------------------------------
    # Peak normalize to ~-1 dBFS so the dynamic range fed to Whisper is
    # consistent regardless of how loud the demucs vocals stem came out.
    peak = float(np.max(np.abs(vocals_audio))) if vocals_audio.size else 0.0
    if peak > 1e-6:
        vocals_audio = (vocals_audio / peak * 0.89).astype(np.float32)

    # 80 Hz high-pass — strips sub-bass rumble that confuses VAD without
    # touching vocal formants (which start above ~250Hz).
    if sr > 200:
        b, a = butter(2, 80.0 / (sr / 2), btype="high")
        vocals_audio = filtfilt(b, a, vocals_audio).astype(np.float32)

    # Slow-audio trick: stretch the vocals to 0.85x speed (15% slower) before
    # Whisper sees them. Whisper is empirically more accurate on slower speech
    # — longer dwell time per phoneme means more reliable recognition on the
    # sung melismas and quieter syllables that the speech-trained model
    # otherwise mangles. Word timestamps are scaled back below.
    STRETCH_RATE = 0.85
    if STRETCH_RATE != 1.0:
        vocals_audio = librosa.effects.time_stretch(
            vocals_audio, rate=STRETCH_RATE
        ).astype(np.float32)

    # Resample to Whisper's native 16kHz.
    target_sr = 16000
    if sr != target_sr:
        vocals_audio = librosa.resample(
            vocals_audio, orig_sr=sr, target_sr=target_sr
        )
    vocals_audio = vocals_audio.astype(np.float32)

    # Decoding params reverted to faster-whisper defaults after observing that
    # the "tighter" tweaks (beam_size=8, condition_on_previous_text=False,
    # narrower temperature ladder, custom VAD min_silence) made larger models
    # noticeably worse on sung material. The defaults are well-tuned for
    # Whisper as shipped; only initial_prompt is layered on as a user knob.
    segments, _info = model.transcribe(
        vocals_audio,
        word_timestamps=True,
        vad_filter=True,
        initial_prompt=initial_prompt or None,
    )

    # Whisper sees the stretched timeline; multiply by STRETCH_RATE to get
    # back to the original (real-time) timestamps the rest of the pipeline
    # expects.
    words: List[LyricWord] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            # whisper sometimes emits leading whitespace on the first word
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


# ---------------------------------------------------------------------------
# Gemini lyric transcription (cloud)
# ---------------------------------------------------------------------------


_GEMINI_FAILURE_LOG = Path(__file__).parent / "logs" / "gemini-failures.log"


def _log_gemini_failure(kind: str, exc: BaseException) -> None:
    """Append a Gemini failure with full traceback to backend/logs/gemini-failures.log.

    The terminal one-liner at the failure site is useful in the moment;
    this file is what we read later to figure out *why* an old run fell
    back. Best-effort — if the write itself fails we swallow it, so a
    logging glitch never masks the original error.
    """
    import datetime
    import traceback
    try:
        _GEMINI_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _GEMINI_FAILURE_LOG.open("a", encoding="utf-8") as f:
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            f.write(
                f"\n=== {ts} {kind} {type(exc).__name__}: {exc} ===\n"
            )
            f.write(traceback.format_exc())
    except Exception as log_exc:
        print(f"[gemini-failure-log] could not write log: {log_exc}")


def _get_genai_client():
    """Lazy-init the Google Gen AI client. Raises if GEMINI_API_KEY is missing."""
    global _genai_client
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is not set")
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def transcribe_with_gemini(
    vocals_audio: np.ndarray, sr: int, initial_prompt: str = "",
    model_override: Optional[str] = None,
) -> List[LyricWord]:
    """
    Cloud-based lyric transcription via Gemini. Multimodal, much better than
    Whisper on sung material because the training data is broader.

    Raises on any failure (missing key, network, refusal, malformed JSON,
    schema mismatch) so the caller can transparently fall back to Whisper.
    """
    import io
    import json
    import soundfile as sf
    from google.genai import types as genai_types

    client = _get_genai_client()

    # Encode the vocals stem as 16-bit PCM WAV in memory. Gemini accepts inline
    # audio up to ~20MB; a multi-minute mono 16kHz stem is well under that.
    buf = io.BytesIO()
    sf.write(buf, vocals_audio, sr, format="WAV", subtype="PCM_16")
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

    response = client.models.generate_content(
        model=model_override if model_override in _GEMINI_MODELS else _GEMINI_MODEL,
        contents=[audio_part, prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    raw = (response.text or "").strip()
    if not raw:
        raise ValueError("Gemini returned empty response")

    # Defensive: strip code fences if the model wrapped its JSON anyway.
    if raw.startswith("```"):
        raw = raw.strip("`")
        # remove possible "json" language tag on the first line
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1 :]
        raw = raw.strip("`").strip()

    def _diag(reason: str) -> ValueError:
        """Wrap a parse failure with a snippet of the raw response so the
        backend log shows exactly what Gemini sent back."""
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
            # one level of nesting (e.g. {"data": {"words": [...]}})
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


# ---------------------------------------------------------------------------
# Gemini chord identification (cloud, hybrid mode)
# ---------------------------------------------------------------------------

# Flat → sharp equivalence map for normalizing chord roots returned by Gemini.
_FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B",  "Fb": "E",   # rare but defensible
}

# Recognized quality suffixes — must match what _BASE_TEMPLATES knows.
_KNOWN_QUALITIES = {"", "m", "7", "maj7", "m7", "sus2", "sus4", "dim", "aug"}


def _infer_better_key(
    reported_key: str, chord_segments: List[ChordSegment]
) -> Tuple[str, Optional[str]]:
    """
    Verify or correct the reported key using three independent chord-timeline
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

    When the inferred root matches the reported root, we leave the key
    alone — Gemini's mode call gets the benefit of the doubt because the
    timeline alone can't always distinguish "i" from "I" without melodic
    information we don't have.

    Returns (final_key, original_if_overridden_else_None).
    """
    import re

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
        # Only pure minor flavors count; maj7/dim/aug/etc. are not "minor".
        is_minor = qual == "m" or qual == "m7"
        return root, is_minor

    # Track major-vs-minor time per root for downstream mode decision.
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

    # Same root as reported — trust the reported mode (Gemini hears subtleties
    # the timeline can't distinguish).
    if inferred_root == reported_root:
        return reported_key, None

    # Different root, but signals are weak — keep the reported answer.
    # Threshold 4.0 requires either:
    #   - first + last chords agreeing on the same root (2.0 + 2.0 = 4.0), OR
    #   - all three signals agreeing on the same root (2.0 + 2.0 + 1.5 = 5.5)
    # A single first/last chord plus most-played (3.5) is NOT enough — when
    # the start and end disagree, the song probably isn't conclusively in
    # the candidate key, so trust the model.
    if inferred_score < 4.0:
        return reported_key, None

    # Strong signals for a different root — override. Mode comes from whichever
    # quality of the inferred root has more total time.
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
    """
    Return the 6 most-common diatonic chords for a major or minor key
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
        # Intervals from tonic for I, ii, iii, IV, V, vi
        intervals = [0, 2, 4, 5, 7, 9]
        qualities = ["", "m", "m", "", "", "m"]
    elif mode == "minor":
        # Intervals from tonic for i, III, iv, v, VI, VII (natural minor)
        intervals = [0, 3, 5, 7, 8, 10]
        qualities = ["m", "", "m", "m", "", ""]
    else:
        return []
    return [f"{PITCH_NAMES[(root_idx + i) % 12]}{q}" for i, q in zip(intervals, qualities)]


def _sanitize_chord_label(label: str) -> Optional[str]:
    """
    Normalize a free-form chord label (as returned by Gemini) into our
    internal shorthand. Returns None if the label can't be parsed or uses
    a quality we don't have a template for.
    """
    import re

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

    # Normalize flats to sharps so we match our PITCH_NAMES indexing.
    root = _FLAT_TO_SHARP.get(root, root)
    if root not in PITCH_NAMES:
        return None

    # Normalize a few common alias spellings before checking the allowlist.
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
    """
    Send the harmonic audio to Gemini and return:
        {"key": "<str or None>", "chords": ["D", "A", "Em", "G", ...]}

    Raises on any failure (missing key, network, refusal, malformed JSON,
    or sanitization leaving < 2 valid chords) so the caller falls back to
    full-vocabulary local recognition.
    """
    import io
    import json
    import soundfile as sf
    from google.genai import types as genai_types

    client = _get_genai_client()

    # 16kHz mono keeps the payload tiny (~4MB for a 4-min track) and is
    # plenty for chord identification — we're not asking Gemini for fine-
    # grained timing.
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

    response = client.models.generate_content(
        model=model_override if model_override in _GEMINI_MODELS else _GEMINI_MODEL,
        contents=[audio_part, prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
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


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------


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
    """
    Most-likely state path under a simple HMM where the only transition
    rule is "staying is cheaper than moving" (self_bonus applied on the
    diagonal of the transition matrix). Penalises one-frame flickers like
    E→Em→E without preventing genuine chord changes — a real chord change
    only has to beat the previous chord by `self_bonus` in Pearson score.

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


def _custom_templates_from_labels(labels: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Build a Pearson-ready template matrix for an arbitrary list of chord
    labels (e.g. ["D", "A", "Em", "G"]). Labels whose quality we don't have
    a known bit-pattern for are silently dropped.

    Returns (kept_labels, centered_normalized_matrix). Raises ValueError if
    fewer than 2 labels survived filtering.
    """
    import re

    # The extended set contains every quality we know — use it as the lookup.
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


def extract_chords(
    y_harm: np.ndarray,
    sr: int,
    duration: float,
    tuning: float,
    vocabulary: str = "triads",
    allowed_chord_labels: Optional[List[str]] = None,
) -> List[ChordSegment]:
    """
    Chord recognition via chroma_cqt + Pearson-correlation template matching,
    smoothed with Viterbi over a self-transition-biased HMM.

    Pearson (not cosine) is critical: with cosine similarity, a noisy or
    near-uniform chroma scores ~equally against every template, so C major
    wins by tiebreak. Pearson subtracts the mean first, so uniform chroma
    scores ~0 against everything.

    Viterbi collapses E/Em-style flickers caused by two guitars sharing
    notes — staying on the previous chord gets a +self_bonus reward, so a
    competing chord has to beat it by at least that margin to flip.

    When `allowed_chord_labels` is non-empty, the template set is restricted
    to those chords (hybrid mode — Gemini identifies the chord set, we do
    the timing). Raises ValueError if fewer than 2 labels survive validation.
    """
    if allowed_chord_labels:
        labels_for_state, template_z = _custom_templates_from_labels(allowed_chord_labels)
    else:
        if vocabulary not in _TEMPLATES:
            vocabulary = "triads"
        labels_for_state, template_z = _TEMPLATES[vocabulary]

    hop = 2048  # ~93ms per frame at sr=22050
    chroma = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, hop_length=hop, bins_per_octave=36, tuning=tuning
    )
    chroma = median_filter(chroma, size=(1, 5))  # ~465ms smoothing

    chroma_z = _center_and_normalize(chroma, axis=0)        # (12, n_frames)
    sims = template_z @ chroma_z                            # (n_templates, n_frames)

    # Treat the silence threshold itself as a state: any frame where no chord
    # template beats this floor will route through the N state.
    N_FLOOR = 0.2
    n_emission = np.full((1, sims.shape[1]), N_FLOOR)
    emission = np.vstack([sims, n_emission])

    # 0.5 ≈ "a competing chord needs to score 0.5 higher in Pearson r
    # to overcome the cost of switching." Tuned for rhythm-guitar/bass-
    # anchored detection — at this stickiness the matcher only flips when
    # the song actually moves to a different chord, not when chroma noise
    # briefly favors a neighbor.
    path = _viterbi(emission, self_bonus=0.5)

    n_state_idx = len(labels_for_state)  # last index = N
    labels = [
        "N" if i == n_state_idx else labels_for_state[i]
        for i in path.tolist()
    ]

    frame_times = librosa.frames_to_time(
        np.arange(len(labels)), sr=sr, hop_length=hop
    )

    # Build raw segments at label-change boundaries
    raw: List[tuple] = []
    seg_start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            raw.append((float(frame_times[seg_start]), float(frame_times[i]), labels[i - 1]))
            seg_start = i
    raw.append((float(frame_times[seg_start]), duration, labels[-1] if labels else "N"))

    # Single-pass merge: absorb short segments into predecessor; also merge
    # any two adjacent segments that share the same chord label.
    # ~1 bar at 144 BPM is 1.67s; the threshold lets segments that span at
    # least a half-bar survive, so we match how a rhythm guitarist actually
    # plays (chord per bar / half-bar) rather than the matcher's flicker.
    MIN_DUR = 1.0
    merged: List[tuple] = []
    for start, end, label in raw:
        if not merged:
            merged.append((start, end, label))
            continue
        prev_start, prev_end, prev_label = merged[-1]
        if prev_label == label or (end - start) < MIN_DUR:
            # Extend previous segment (absorb or merge same chord)
            merged[-1] = (prev_start, end, prev_label)
        else:
            merged.append((start, end, label))

    return [ChordSegment(start=s, end=e, label=l) for s, e, l in merged]


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------


def detect_sections(
    y_harm: np.ndarray, sr: int, duration: float, beat_frames: np.ndarray
) -> List[Section]:
    """
    Best-effort song structure: boundary detection on beat-synced chroma,
    KMeans cluster the inter-boundary segments, then heuristically name the
    clusters as Verse / Chorus / Bridge / Intro / Outro.

    Returns segments ordered by start time. Adjacent segments sharing the
    same emitted label are merged. Empty list if there aren't enough beats
    to cluster meaningfully.
    """
    from sklearn.cluster import KMeans

    n_beats = len(beat_frames)
    if n_beats < 16:
        return []

    chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr)
    beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    # (12, n_beats); beats[i] is the time of beat i

    # Boundaries returned as beat-indices. Choose k roughly proportional to
    # song length but capped so very short songs still get a few sections.
    k_segs = max(3, min(8, n_beats // 16))
    bounds = librosa.segment.agglomerative(beat_chroma, k=k_segs)
    bounds = np.asarray(bounds, dtype=int)

    # Ensure 0 and end appear as edges, then sort/dedupe.
    edges = np.unique(np.concatenate([[0], bounds, [n_beats]]))
    # Sanity: at least two edges so we have at least one segment.
    if len(edges) < 2:
        return []

    # Build (start_beat, end_beat) segments and their average chroma vector.
    seg_feats = []
    seg_ranges = []
    for i in range(len(edges) - 1):
        s, e = int(edges[i]), int(edges[i + 1])
        if e <= s:
            continue
        feat = beat_chroma[:, s:e].mean(axis=1)
        seg_feats.append(feat)
        seg_ranges.append((s, e))

    if len(seg_feats) < 2:
        return []

    feats = np.stack(seg_feats)  # (n_segments, 12)
    n_clusters = min(4, len(seg_feats))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    cluster_ids = km.fit_predict(feats).tolist()

    # Frequency of each cluster across segments
    counts: dict = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1

    sorted_clusters = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    # Verse/Chorus naming only makes sense when there are TWO clusters that
    # each repeat. If the song is chordally uniform (one cluster owns most of
    # the timeline, everything else is one-off), calling the recurring cluster
    # "Chorus" is misleading — fall back to neutral "Section A/B/…" labels.
    has_two_recurring = (
        sorted_clusters[0][1] >= 2
        and len(sorted_clusters) > 1
        and sorted_clusters[1][1] >= 2
    )

    if has_two_recurring:
        chorus_id = sorted_clusters[0][0]
        verse_id = sorted_clusters[1][0]
        # Pop convention: the chord pattern that appears EARLIEST is usually
        # the verse; the second distinct recurring pattern is the chorus.
        # Swap if the "chorus" candidate first appears before the "verse".
        first_chorus = next(i for i, c in enumerate(cluster_ids) if c == chorus_id)
        first_verse = next(i for i, c in enumerate(cluster_ids) if c == verse_id)
        if first_chorus < first_verse:
            chorus_id, verse_id = verse_id, chorus_id

        base_names: List[str] = []
        for idx, cid in enumerate(cluster_ids):
            if cid == chorus_id:
                base_names.append("Chorus")
            elif cid == verse_id:
                base_names.append("Verse")
            elif idx == 0:
                base_names.append("Intro")
            elif idx == len(cluster_ids) - 1:
                base_names.append("Outro")
            else:
                base_names.append("Bridge")
    else:
        # Degenerate clustering: map each cluster id to a letter by frequency.
        letter_for_cluster = {
            cid: chr(ord("A") + i) for i, (cid, _) in enumerate(sorted_clusters)
        }
        base_names = [
            f"Section {letter_for_cluster[cid]}" for cid in cluster_ids
        ]
        # Still honour Intro/Outro when the first/last segments are singletons.
        if counts[cluster_ids[0]] == 1:
            base_names[0] = "Intro"
        if counts[cluster_ids[-1]] == 1:
            base_names[-1] = "Outro"

    # Number labels that repeat (Verse 1, Verse 2; Chorus, Chorus 2; etc.).
    # Unique labels (Intro, Outro) usually stay un-numbered.
    counts_by_label: dict = {}
    for n in base_names:
        counts_by_label[n] = counts_by_label.get(n, 0) + 1

    seen: dict = {}
    final_labels: List[str] = []
    for n in base_names:
        if counts_by_label[n] == 1:
            final_labels.append(n)
        else:
            seen[n] = seen.get(n, 0) + 1
            final_labels.append(f"{n} {seen[n]}")

    # Convert beat-indices to seconds and pair with labels.
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    raw: List[tuple] = []
    for (s_beat, e_beat), label in zip(seg_ranges, final_labels):
        start_t = float(beat_times[s_beat]) if s_beat < len(beat_times) else 0.0
        end_t = (
            float(beat_times[e_beat])
            if e_beat < len(beat_times)
            else duration
        )
        # Clip to song bounds
        start_t = max(0.0, min(start_t, duration))
        end_t = max(start_t, min(end_t, duration))
        raw.append((start_t, end_t, label))

    # Merge adjacent segments sharing the same emitted label (after the
    # numbering pass these are identical labels only if numbering collapsed,
    # but be safe).
    merged: List[tuple] = []
    for s, e, label in raw:
        if merged and merged[-1][2] == label:
            ps, _, pl = merged[-1]
            merged[-1] = (ps, e, pl)
        else:
            merged.append((s, e, label))

    return [Section(start=s, end=e, label=l) for s, e, l in merged]


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
    if not file.filename:
        raise HTTPException(400, "No filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}:
        raise HTTPException(400, f"Unsupported format: {suffix}")

    if transcribe_lyrics and not separate_stems:
        # Whisper on a full mix produces garbage; the isolated vocals stem
        # is what makes transcription tolerable on songs with backing music.
        _progress["stage"] = "idle"
        _progress["pct"] = 0
        raise HTTPException(
            400, "transcribe_lyrics requires separate_stems=true"
        )

    # Read the uploaded bytes once so we can compute a cache key before
    # spending any compute. Sync read because we're a sync handler.
    body_bytes = file.file.read()

    # Cache lookup. Same audio + same params → instant return. Different
    # params (e.g., toggling Gemini Pro vs Flash) → different key.
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
        # Settle the progress bar immediately so the UI's poller sees done.
        _progress["stage"] = "done"
        _progress["pct"] = 100
        print(f"[cache] hit ({len(body_bytes)} bytes) — returning cached result")
        return AnalysisResult(**cached)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(body_bytes)
        tmp_path = tmp.name

    # Declare which stages will run so the progress bar weights are right
    # for this particular request. The "gemini" stage covers the parallel
    # chord_id + lyric-transcription Gemini calls. "lyrics_whisper" is added
    # up front only when Whisper is the explicit provider; the fallback path
    # (Gemini failed → Whisper) adds it dynamically via _progress_extend.
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
        # the wall time of the shorter of the two. The local pipeline
        # (tuning_key, extract_chords, sections) continues sequentially after.
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

        # Augment Gemini's chord set with two layers of safety net:
        #   (1) Parallel major/minor of every Gemini chord (D ↔ Dm, E ↔ Em, …).
        #       Robust against the common Gemini failure of swapping a chord's
        #       major/minor when the third is unclear in the audio.
        #   (2) Diatonic chords of Gemini's key, when it returned one. Covers
        #       the case where Gemini omits a chord that's in the song (e.g.
        #       returns "D A G" for a song that also uses Em).
        # The UI still shows the original gemini_chord_set so the user sees
        # exactly what Gemini reported.
        allowed_for_matcher: Optional[List[str]] = None
        if gemini_chord_set:
            allowed_for_matcher = list(gemini_chord_set)
            # Parallel major/minor pass (cheap; just doubles the triad count)
            for c in list(allowed_for_matcher):
                # Plain major: "D" → parallel minor "Dm"
                if c in PITCH_NAMES and f"{c}m" not in allowed_for_matcher:
                    allowed_for_matcher.append(f"{c}m")
                # Plain minor: "Em" → parallel major "E"
                elif c.endswith("m") and not c.endswith(("dim", "aug")):
                    root = c[:-1]
                    if root in PITCH_NAMES and root not in allowed_for_matcher:
                        allowed_for_matcher.append(root)
            # Diatonic pass
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

        # Tonic sanity check from chord-timeline signals (first chord, last
        # chord, most-played chord). Catches cases like "song reported as
        # E minor but actually starts/ends/lives on D" by majority vote
        # across the three signals. Conservative — only overrides when the
        # inferred root differs from the reported root AND has strong votes.
        key, _key_was = _infer_better_key(key, chords)

        _progress_enter("sections")
        sections = detect_sections(y_harm, sr, duration, beat_frames)
        _progress_complete("sections")

        # Whisper path: fires either as the primary provider (lyric_provider=
        # whisper) or as the fallback when Gemini lyric transcription failed
        # above. The fallback case wasn't included in the initial stage list,
        # so we extend the progress pool dynamically before entering.
        if (
            transcribe_lyrics
            and vocals_audio is not None
            and vocals_sr is not None
            and lyrics is None
        ):
            if want_lyric_gemini:
                # Gemini failed above; add the fallback weight now.
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
        # Stash for next time the same audio + params land. model_dump() gives
        # a JSON-friendly dict that reconstructs cleanly. Skip when Gemini
        # fell back — caching a fallback would short-circuit every future
        # retry, even after Google's outage clears.
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
