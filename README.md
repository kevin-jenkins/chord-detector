# Chord Detector

A local-only chord detection web app for Windows 11. Upload a song, get a
synced chord timeline with transpose, capo suggestions, and ChordPro export.
Everything runs on `localhost` — audio never leaves your machine.

## Stack

- **Backend**: FastAPI + librosa (beats, key) + Chordino via `chord-extractor` (chords)
- **Frontend**: Next.js 14 + wavesurfer.js

## One-time setup

### 1. Install the Vamp Plugin Pack (Windows installer)

Chordino is a Vamp plugin written in C++. Download and run the installer from:

https://www.vamp-plugins.org/pack.html

Pick **Windows (64-bit)** — `Vamp Plugin Pack Installer 1.0.exe`. Accept the
defaults. This installs Chordino's `.dll` into `C:\Program Files\Vamp Plugins`.

`chord-extractor` will find this automatically on Windows. If it doesn't, set
the env var: `VAMP_PATH=C:\Program Files\Vamp Plugins`.

### 2. Install FFmpeg (so librosa can read mp3/m4a/etc.)

```powershell
winget install Gyan.FFmpeg
```

Then close and reopen PowerShell so `ffmpeg.exe` is on PATH.

### 3. Backend

Requires Python 3.11 (3.12 also works; 3.13 may have wheel gaps for some deps).

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Frontend

Requires Node.js 20+.

```powershell
cd frontend
npm install
```

## Running

Two terminals.

**Terminal 1 — backend:**
```powershell
cd backend
.venv\Scripts\Activate.ps1
python main.py
```
Listens on `http://127.0.0.1:8000`. Hit `http://127.0.0.1:8000/health` to confirm
Chordino is wired up — the `vamp_path` field should point at your install.

**Terminal 2 — frontend:**
```powershell
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## Using it

1. Drop an audio file onto the upload zone
2. Wait 10-30s for analysis (longer for full albums)
3. The big chord above the strip is the current chord; click any chord block
   to seek there
4. Use ± to transpose; "Capo N → play in X" suggests an easy guitar key
5. **Export ChordPro** dumps a timed text chart

## How it works

```
audio file
   │
   ├──► librosa ──► beats, BPM, chroma → Krumhansl-Schmuckler key estimate
   │
   └──► chord-extractor → Chordino (NNLS-Chroma Vamp plugin)
                              │
                              └─► [(start, end, "C:maj"), ...]
                                   parsed into {root, quality, bass}
                                   on the frontend, transposable
```

Chordino's vocabulary includes triads, 7ths, sus, dim, aug, and slash chords.
Accuracy is strong on clear pop/rock arrangements (~75-85% frame accuracy on
typical material) and weaker on dense mixes or distorted electric guitar with
heavy sustain.

## Improving accuracy

For tougher material — like the kind of layered indie rock arrangements where
distorted guitar muddies the chroma — run stem separation first:

```powershell
pip install demucs
demucs --two-stems=other "song.mp3"
# then upload separated/htdemucs/song/other.wav instead of the original
```

The `other` stem (everything except vocals) tends to give noticeably better
chord detection. You could wire demucs into the backend as a checkbox; left
out of v1 to keep the install lean (demucs pulls ~2GB of model weights).

## Swapping in a better model

The chord detection layer is one function in `backend/main.py` (`extract_chords`).
To upgrade to a transformer-based model (BTC, ChordMini), replace the body of
that function — the rest of the pipeline is agnostic to which recognizer
produces the `(start, end, label)` tuples.
