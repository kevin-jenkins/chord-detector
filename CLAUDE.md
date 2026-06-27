# Chord Detector — Project Brief

This file is read automatically by Claude Code at the start of each session.
It captures decisions made in the planning conversation so you don't need to
re-explain context.

## What this is

A local-only chord detection web app, conceptually similar to Chord AI but
running entirely on the user's Windows 11 machine. User uploads an audio file,
gets back a synced chord timeline with transpose, capo suggestions, and
ChordPro export.

The user is a musician (member of Megayacht, a Chapel Hill indie rock band)
with a strong software engineering background, including JUCE/VST development.
They asked for this because they want chord detection that doesn't require
uploading their unreleased material to a third party.

**Privacy invariant — partial exception**: chord detection, beat tracking,
key estimation, section detection, and stem separation all run fully locally.
Lyric transcription has two providers, user-selectable per song:
- **Whisper (local)** — audio never leaves the machine.
- **Gemini (cloud)** — vocals stem is sent to Google AI Studio for higher
  accuracy on sung material. Used by default because local Whisper is
  unusably inaccurate on this user's unreleased indie-rock vocals. Gemini
  failures auto-fall-back to local Whisper-small.

## Stack & key decisions

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI on `127.0.0.1:8000` | Simple, async, easy CORS to Next dev server |
| Chord detection | Chordino via `chord-extractor` (Python wrapper around the NNLS-Chroma Vamp plugin) | Better vocabulary than autochord (triads + 7ths/sus/dim/aug/slash), Windows installer exists |
| Beats + key | librosa `beat_track` + Krumhansl-Schmuckler from chroma | Standard, fast, no extra deps |
| Frontend | Next.js 14 App Router + wavesurfer.js | User already uses Next.js for other projects (download.megayachtband.com) |
| Runtime | Native Python venv on Windows 11 | User explicitly chose this over Docker/WSL |
| Chord vocabulary | Triads + 7ths/sus/dim (Chordino default) | Sweet spot for indie rock; jazz vocabulary out of scope for v1 |

## Explicitly out of scope for v1

These were considered and deferred — don't add them without asking:

- **Stem separation (demucs)**: Adds ~2GB of model weights and 30s/song.
  Mentioned in README as a manual preprocessing step. Could become a backend
  checkbox in v2.
- **Guitar/piano chord diagrams**: vexchords integration is a separate rabbit
  hole. v1 just shows chord labels.
- **YouTube URL ingestion**: Not requested. Would require yt-dlp.
- **Lyric transcription**: Not requested. Would require Whisper.
- **Better chord model (BTC / ChordMini transformers)**: `extract_chords()` in
  `backend/main.py` is the swap point. The pipeline is agnostic to which
  recognizer produces `(start, end, label)` tuples.

## Project layout

```
backend/
  main.py              # FastAPI app, one /analyze endpoint
  requirements.txt
frontend/
  src/
    app/
      page.tsx         # main page: upload, waveform, transport, transpose, export
      layout.tsx
      globals.css
    components/
      ChordTimeline.tsx  # big "now/next" + scrollable chord strip
      chord.ts           # parse/format/transpose chord labels
      export.ts          # ChordPro export
      types.ts           # shared TS types
  package.json
  tsconfig.json
README.md              # Windows 11 setup instructions
```

## Run it

Two terminals:

```powershell
# Backend
cd backend
.venv\Scripts\Activate.ps1
python main.py

# Frontend
cd frontend
npm run dev
```

Open http://localhost:3000.

## Known gotchas (Windows 11 specifically)

1. **Vamp Plugin Pack must be installed before pip install.** `chord-extractor`
   looks for the Chordino DLL in `C:\Program Files\Vamp Plugins`. If the
   `/health` endpoint shows `vamp_path: (not set; using system default)` AND
   analysis fails, set `VAMP_PATH` env var explicitly.

2. **FFmpeg must be on PATH** before running the backend, or librosa can't
   decode mp3/m4a. `winget install Gyan.FFmpeg` then restart the shell.

3. **numpy install order**: `chord-extractor`'s setup.py imports numpy before
   installing it. If `pip install -r requirements.txt` fails, run
   `pip install numpy==1.26.4` first, then retry.

4. **Python 3.13 wheels**: Some deps (notably `vamp`) may not have 3.13 wheels
   yet. Use Python 3.11 or 3.12.

## Likely next things the user will want

In rough priority order — confirm before building:

1. **Test it on Megayacht material** ("20B", "Impossible"). Accuracy on full
   mixes with distorted guitar is the obvious risk; if it's rough, demucs
   preprocessing is the first fix.
2. **Demucs integration** as a backend toggle ("separate stems first").
3. **Guitar chord diagrams** under each chord in the timeline.
4. **Section detection** (verse/chorus/bridge) — librosa has `segment.agglomerative`.
5. **Loop/punch-in** for transcription practice (set start/end, loop, slow down).
6. **Save/load analyses** so you don't re-analyze the same file twice.

## Code style notes

- Prose comments over heavy JSDoc. The codebase is small enough that
  variable names + short comments do the job.
- Backend: type hints on public functions, Pydantic models for API I/O.
- Frontend: TypeScript strict mode, no external state lib (useState is fine
  at this size).
- Keep the backend stateless. No DB, no on-disk cache of analyses in v1.
