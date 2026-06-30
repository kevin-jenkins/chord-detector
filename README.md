# Chord Detector

A cross-platform chord detection web app (Windows, macOS, Linux). Drop a song in, get back a synced
chord timeline, lyric transcription, section structure, transpose / capo
suggestions, and exports in ChordPro or Ultimate Guitar–style tab.

Built around a hybrid of local DSP and Google Gemini: stem separation, beat
tracking, key estimation, and template-matched chord recognition run locally;
chord *identification* and lyric *transcription* optionally use Gemini (cloud)
for substantially better quality on hard material — out-of-the-box Whisper
loses to Pro on heavy/melismatic indie-rock vocals by a wide margin.

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI on `127.0.0.1:8000` |
| Stem separation | demucs `mdx_extra_q` (quantized MDX-Net, CPU) |
| Beat / BPM | librosa `beat_track` |
| Key estimate | Krumhansl-Schmuckler chroma profile + timeline-tonic correction |
| Chord recognition (local) | chroma_cqt → Pearson template matching → Viterbi |
| Chord identification (cloud) | Gemini 2.5 Flash or Pro — returns the chord set + key |
| Lyric transcription | Gemini (default) with faster-whisper `small` as automatic fallback |
| Section detection | librosa boundary clustering + KMeans on segment-averaged chroma |
| Frontend | Next.js 14 + wavesurfer.js |
| Caching | Two-tier: result cache by `sha256(audio + params)`, stems sub-cache by `sha256(audio)`. Survives backend restarts. |

## Setup

Requires **Python 3.11** (3.12 also fine), **Node 20+**, and **FFmpeg** on PATH.

| Platform | FFmpeg | venv activate |
|---|---|---|
| Windows (PowerShell) | `winget install Gyan.FFmpeg` then restart the shell | `.venv\Scripts\Activate.ps1` |
| macOS | `brew install ffmpeg` | `source .venv/bin/activate` |
| Debian/Ubuntu | `sudo apt install ffmpeg` | `source .venv/bin/activate` |
| Fedora/RHEL | `sudo dnf install ffmpeg` | `source .venv/bin/activate` |

### Backend

```bash
cd backend
python -m venv .venv
# activate per the table above
pip install -r requirements.txt
```

### GPU acceleration (optional, big win)

Demucs auto-detects the best device on the first analysis after a restart.
You'll see one of these lines in the backend log:

- `Demucs device: CUDA (...)` — NVIDIA GPU on Linux/Windows. Roughly **10× faster** than CPU.
- `Demucs device: MPS (Apple Silicon, CPU fallback enabled)` — M1/M2/M3 Mac. Roughly **5–8× faster** than CPU.
- `Demucs device: CPU` — universal fallback.

CUDA requires a matching `torch` build (`pip install torch --index-url https://download.pytorch.org/whl/cu121` after the regular install, picking the cuXXX that matches your CUDA toolkit). MPS works out of the box on Apple Silicon with the standard `torch` wheel from PyPI.

First run will download:
- demucs weights (`mdx_extra_q`, ~50 MB) to `~/.cache/torch/hub`
- faster-whisper weights (`small`, ~480 MB) to `~/.cache/huggingface/hub` — only fetched the first time Whisper is actually needed

### Frontend

```bash
cd frontend
npm install
```

### Gemini API key

Both the chord-identification and lyric-transcription paths use Google's
Gemini API by default. Get a free key from
[Google AI Studio](https://aistudio.google.com/app/apikey) and drop it into
`backend/.env`:

```ini
# backend/.env  (gitignored)
GEMINI_API_KEY=AIza...your_key...

# Optional model override. Default is gemini-2.5-flash.
# Use gemini-2.5-pro for higher quality on hard vocal material;
# the free tier doesn't include Pro, so you'll need a billing-enabled
# Cloud project for Pro requests.
# GEMINI_MODEL=gemini-2.5-pro
```

**Without a key**, the app still runs — both Gemini paths will silently fall
back to local-only (Pearson + Viterbi chord matching against the full template
set, and Whisper-small for lyrics). You won't see any errors; the response
fields `chord_provider_used` / `lyric_provider_used` will show
`local-fallback` / `whisper-fallback`.

**Billing for Pro.** The free tier currently allocates *zero* quota for
Gemini 2.5 Pro — your first Pro request fires `RESOURCE_EXHAUSTED` with
`limit: 0`. Enable billing on the Cloud project tied to your AI Studio
account and Pro starts working. Cost is roughly **$0.03 per song** (both
Gemini calls combined). Set a budget alert in the Cloud console if you want
peace of mind. As a bonus, the paid tier excludes your audio from being
used for Google's model training, which the free tier does not.

## Running

Two terminals.

```bash
# Terminal 1 — backend
cd backend
# activate venv:
#   Windows:    .venv\Scripts\Activate.ps1
#   macOS/Linux: source .venv/bin/activate
python main.py
```

```bash
# Terminal 2 — frontend
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## Using it

Drop an audio file on the upload zone. Default settings analyze with stem
separation + Gemini for both chord ID and lyrics on Pro. Expected wall-clock:

- **First run on a new file**: ~3–5 minutes (demucs is most of it).
- **Re-upload of the same file with same settings**: ~0.5 s (result cache hit).
- **Same audio, different knob** (e.g., switch Pro → Flash, change vocabulary): ~70 s
  (stems sub-cache hits — demucs is skipped, only Gemini reruns).

What the controls do:

| Control | Effect |
|---|---|
| **Separate stems first** | Run demucs to drop drums and isolate vocals before chord ID and lyric transcription. Default on. Off → run HPSS as a poor-man's substitute. |
| **Transcribe lyrics** | Send the vocals stem to your chosen lyric provider. Default on. |
| **Chord vocabulary** | Triads / Triads + 7ths / Extended (sus, dim, aug, 7ths). Default triads. |
| **Chord detection** | Hybrid (Gemini identifies the chord set + key, local Pearson + Viterbi handles timing) or Local only. Hybrid is dramatically better on the user's material. |
| **Gemini model** | 2.5 Flash (fast, free tier) or 2.5 Pro (slower, billing required). Pro is meaningfully better on dense mixes. |
| **Provider** (lyrics) | Gemini (cloud) or Whisper (local). |
| **Hint** | Free-text passed into the Gemini prompts as context — useful for band names, song titles, genre hints that bias the model toward the right vocabulary. |
| **Transpose ± / Reset** | Shift the whole chart up/down semitones. |

When analysis finishes, the meta row shows:
- Key (corrected by a heuristic that compares Gemini's reported tonic against the chord-timeline first/last/most-played signals — if Gemini calls a song E minor but the timeline starts/ends/lives on D, it auto-flips to D major)
- BPM, total chord count, processing time
- Chord set returned by Gemini (if the hybrid path ran)
- Capo suggestion if the song is in a "hard" guitar key

## How it works

```
audio file (uploaded once, hashed for cache key)
   │
   ├──► librosa.load + beat_track  (BPM, beat timestamps)
   │
   ├──► demucs (mdx_extra_q)       drums, bass, other, vocals
   │      │   bass + other ─────┐  vocals ─────┐
   │      │                     │              │
   │      ▼                     │              │
   │   ↓ stems cache by audio hash             │
   │                            │              │
   │                            ▼              ▼
   │              [PARALLEL Gemini calls]
   │                            │              │
   │   identify_chords_with_gemini      transcribe_with_gemini
   │   → {key, chords[]}                 → [{word, start, end}, …]
   │                            │              │
   ├──► librosa.estimate_tuning  │              │
   ├──► Krumhansl-Schmuckler key │              │
   │                            ▼              │
   ├──► extract_chords          │              │
   │    chroma_cqt → Pearson → Viterbi         │
   │    (templates restricted to Gemini's      │
   │     chord set + parallel maj/min +        │
   │     diatonic expansion of Gemini's key)   │
   │                            │              │
   │      key flip-check using first/last/most-played
   │                            │              │
   ├──► detect_sections                         │
   │    boundary detection + KMeans clustering  │
   │                            │              │
   │                            ▼              ▼
   └──────► AnalysisResult { key, bpm, chords, sections, lyrics,
                              gemini_chord_set, gemini_key,
                              chord_provider_used, lyric_provider_used }
                              │
                              ▼
              cached to disk by sha256(audio + all knobs)
```

**Why hybrid is good.** Local Pearson + Viterbi alone fights several hard
problems on indie-rock material: chord-uniform songs (D-A-Em-G all the way
through) blow up Krumhansl-Schmuckler's key estimate; two simultaneous
guitars muddy the chroma and create E↔Em flicker; the matcher has to choose
from 24/60/108 templates depending on vocabulary. **Gemini's broader music
training nails the chord set and key on tracks that defeat the matcher**;
once Gemini constrains the template pool, local Pearson + Viterbi handles
the timing precisely.

**Why Gemini for lyrics.** Whisper was trained on speech, not singing.
Bigger Whisper models don't help on out-of-distribution audio — they just
hallucinate more confidently. Gemini's broader audio training gives
substantially better word accuracy on sung material, especially with
melismatic vocals or dense mixes. On clean speech-like vocals the
difference is smaller and Whisper-small is a fine local fallback.

**Whisper auto-fallback.** Any Gemini failure (no API key, rate limit,
network blip, refusal, malformed JSON) silently falls back to Whisper-small.
You always get lyrics; the response flags whether they came from Gemini or
the fallback.

## Privacy

The CLAUDE.md "audio never leaves the machine" guarantee no longer strictly
holds. To be explicit:

- **Stem separation, beat tracking, key estimation, chord recognition,
  section detection** all run fully locally.
- **Chord identification** (hybrid mode) sends the demucs harmonic mix
  (bass + other, 16 kHz mono) to Google's Gemini API.
- **Lyric transcription** (default Gemini path) sends the demucs vocals
  stem (16 kHz mono) to Google's Gemini API.

If your unreleased material can't leave your machine, switch **Chord
detection → Local only** and **Provider → Whisper**, and audio truly stays
on the box. You can also leave `GEMINI_API_KEY` unset — the app will
silently use the local-only path on every request.

Paid-tier Gemini ($billing enabled) excludes your audio from being used
in Google's model training. Free-tier does not.

## Exports

Two formats, both downloaded straight from the browser:

- **`.chordpro.txt`** — real ChordPro: `[D]Lyric word [A]another`, with
  `[Section]` headers and bar-chart fallback for instrumental sections.
- **`.tab.txt`** — Ultimate-Guitar-style: chord names positioned by
  character offset above the lyric they fall on, phrase-aware line
  breaks driven by Gemini's sentence-case capitalization, instrumental
  sections collapsed into `pattern (xN) remainder` form.

Both honor the transpose value at the top of the page; the chord
chart and exports update in lockstep.

## Cache

- **Result cache** — JSON, on disk at `backend/.cache/results/<hash>.json`.
  Key is `sha256(audio_bytes + all knobs)`. Survives backend restart.
- **Stems sub-cache** — in memory, keyed by `sha256(audio_bytes)`. Lets
  you flip Pro ↔ Flash, change vocab, change hint, etc., without
  re-running demucs. Wiped on backend restart.

Wipe the result cache:
```bash
# macOS/Linux
rm -rf backend/.cache

# Windows (PowerShell)
Remove-Item backend\.cache -Recurse -Force
```

## Performance

Where the time goes for a 4-minute song on a modern laptop CPU, with no GPU
acceleration. (These numbers are from the original dev machine — a Windows
laptop whose Intel Iris Xe GPU has no useful demucs path, since demucs uses
STFT/complex tensors that DirectML doesn't support. On a CUDA NVIDIA GPU or
Apple Silicon MPS, demucs is substantially faster — see
[GPU acceleration](#gpu-acceleration-optional-big-win) above.)

| Stage | Time |
|---|---|
| demucs (`mdx_extra_q`, `shifts=0`, `overlap=0.1`) | ~3 min |
| Gemini Pro chord ID + lyric transcription (parallel) | ~60 s |
| librosa load, beats, tuning, chords, sections | ~10 s |
| **Cold total** | **~4 min** |
| **Cache hit (same audio + knobs)** | **~0.5 s** |
| **Stems-cache hit (same audio, different knobs)** | **~70 s** |

## Layout

```
backend/
  main.py              # FastAPI app, all the analysis logic
  requirements.txt
  .env                 # GEMINI_API_KEY (gitignored)
  .cache/results/      # JSON cache, gitignored
frontend/
  src/app/             # Next.js page, layout, icon, styles
  src/components/      # ChordTimeline, types, chord helpers, exporters
CLAUDE.md              # project brief read by Claude Code at session start
```
