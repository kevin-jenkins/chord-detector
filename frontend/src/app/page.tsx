"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import ChordTimeline from "@/components/ChordTimeline";
import ChordEditBar from "@/components/ChordEditBar";
import { AnalysisResult, ChordSegment } from "@/components/types";
import {
  deleteChord, editsStorageKey, insertChord, loadEdits, nudgeToBeat, PENDING_LABEL, relabelChord,
  saveEdits, snapToBeat,
} from "@/components/chordEdits";
import { formatChord, parseChord, transposeChord, transposeKey } from "@/components/chord";
import { exportChordPro, downloadText } from "@/components/export";
import { exportTab } from "@/components/tabExport";

// Must match the port in backend/main.py.
const API = "http://127.0.0.1:8010";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  // Hand corrections. null = showing the detected chords as-is. The undo and
  // redo stacks hold other values of editedChords (null meaning "detected").
  const [editedChords, setEditedChords] = useState<ChordSegment[] | null>(null);
  const [undoStack, setUndoStack] = useState<(ChordSegment[] | null)[]>([]);
  const [redoStack, setRedoStack] = useState<(ChordSegment[] | null)[]>([]);
  const [editing, setEditing] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  // Where a chord is being inserted (seconds), while the edit bar asks which one.
  const [insertAt, setInsertAt] = useState<number | null>(null);
  // editsReady flips once saved edits have been looked up; editsKey stays null
  // if the file couldn't be hashed, and edits then last only this session.
  const [editsKey, setEditsKey] = useState<string | null>(null);
  const [editsReady, setEditsReady] = useState(false);
  const [saveFailed, setSaveFailed] = useState(false);
  const chordInputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [semitones, setSemitones] = useState(0);
  const [separateStems, setSeparateStems] = useState(true);
  const [vocabulary, setVocabulary] = useState<"triads" | "triads_7ths" | "extended">("triads");
  const [transcribeLyrics, setTranscribeLyrics] = useState(true);
  const [lyricProvider, setLyricProvider] = useState<"gemini" | "whisper">("gemini");
  const [whisperModel, setWhisperModel] = useState<"small" | "medium" | "large-v3">("small");
  const [lyricHint, setLyricHint] = useState("");
  const [chordProvider, setChordProvider] = useState<"hybrid" | "local">("hybrid");
  // Second-tier recognizer. "local" uses Pearson+Viterbi (ships by default);
  // "btc" uses the BTC transformer and bypasses the Gemini chord-set constraint.
  // Requires: pip install transformers huggingface_hub.
  const [chordRecognizer, setChordRecognizer] = useState<"local" | "btc">("local");
  // Keep in sync with _GEMINI_MODELS / _GEMINI_LYRIC_MODELS in backend/gemini_client.py.
  const [geminiModel, setGeminiModel] = useState<"gemini-3.5-flash" | "gemini-3.6-flash" | "gemini-3.1-pro-preview">("gemini-3.1-pro-preview");
  // "" = same as geminiModel. 3.8 Flash is lyrics-only: strong at
  // transcription, unreliable at chord identification.
  const [geminiLyricModel, setGeminiLyricModel] = useState<"" | "gemini-3.5-flash" | "gemini-3.6-flash" | "gemini-3.8-flash" | "gemini-3.1-pro-preview">("");
  const [progress, setProgress] = useState(0);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [processingTimeSec, setProcessingTimeSec] = useState<number | null>(null);
  const [volume, setVolume] = useState(1);

  // Lyrics need the vocals stem — keep the two toggles in sync so the user
  // can't end up in an invalid state (which the backend would reject anyway).
  useEffect(() => {
    if (!separateStems && transcribeLyrics) setTranscribeLyrics(false);
  }, [separateStems, transcribeLyrics]);

  // Elapsed-time counter while analysis is running. Starts at 0 each time
  // loading flips true; pauses (and shows the final elapsed time) when loading
  // returns to false until the next request resets it.
  useEffect(() => {
    if (!loading) return;
    setElapsedSec(0);
    const start = Date.now();
    const id = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [loading]);

  // Push volume changes to the live wavesurfer instance so the slider
  // controls playback in real time.
  useEffect(() => {
    waveSurferRef.current?.setVolume?.(volume);
  }, [volume]);

  // Poll /progress while loading. Cancelled on unmount or when loading flips
  // back to false. Sample at 2 Hz — the bar smooths visually via CSS transition.
  useEffect(() => {
    if (!loading) return;
    let cancelled = false;
    setProgress(0);

    const poll = async () => {
      try {
        const r = await fetch(`${API}/progress`);
        if (!r.ok) return;
        const { pct } = await r.json();
        if (!cancelled && typeof pct === "number") setProgress(pct);
      } catch {
        /* swallow — fetch errors during shutdown are expected */
      }
    };

    const id = setInterval(poll, 500);
    poll(); // first sample immediately so the bar moves off zero quickly
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [loading]);

  const waveContainerRef = useRef<HTMLDivElement>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);

  // ---- Upload + analyze --------------------------------------------------

  const analyze = useCallback(async (
    f: File,
    withStems: boolean,
    vocab: string,
    withLyrics: boolean,
    model: string,
    hint: string,
    provider: string,
    chordProv: string,
    geminiM: string,
    geminiLyricM: string,
    recognizer: string,
  ) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setCurrentTime(0);
    setSemitones(0);
    setProcessingTimeSec(null);

    const startedAt = Date.now();
    try {
      const fd = new FormData();
      fd.append("file", f);
      fd.append("separate_stems", withStems ? "true" : "false");
      fd.append("vocabulary", vocab);
      fd.append("transcribe_lyrics", withLyrics ? "true" : "false");
      fd.append("whisper_model", model);
      fd.append("lyric_hint", hint);
      fd.append("lyric_provider", provider);
      fd.append("chord_provider", chordProv);
      fd.append("gemini_model", geminiM);
      fd.append("gemini_lyric_model", geminiLyricM);
      fd.append("chord_recognizer", recognizer);
      const res = await fetch(`${API}/analyze`, { method: "POST", body: fd });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Analysis failed (${res.status}): ${detail}`);
      }
      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Unknown error");
    } finally {
      setProcessingTimeSec((Date.now() - startedAt) / 1000);
      setLoading(false);
    }
  }, []);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      analyze(f, separateStems, vocabulary, transcribeLyrics, whisperModel, lyricHint, lyricProvider, chordProvider, geminiModel, geminiLyricModel, chordRecognizer);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) {
      setFile(f);
      analyze(f, separateStems, vocabulary, transcribeLyrics, whisperModel, lyricHint, lyricProvider, chordProvider, geminiModel, geminiLyricModel, chordRecognizer);
    }
  };

  // ---- Chord corrections -------------------------------------------------

  // Edits are saved per audio file (see editsStorageKey), so a reload or a
  // re-analysis of the same file brings them back.
  useEffect(() => {
    setEditedChords(null);
    setUndoStack([]);
    setRedoStack([]);
    setSelectedIdx(null);
    setInsertAt(null);
    setEditing(false);
    setEditsKey(null);
    setEditsReady(false);
    setSaveFailed(false);
    if (!result || !file) return;
    let cancelled = false;
    editsStorageKey(file)
      .then((key) => {
        if (cancelled) return;
        setEditsKey(key);
        setEditedChords(loadEdits(key, result.duration));
      })
      .catch((err) => console.error("couldn't hash audio for saved edits:", err))
      .finally(() => { if (!cancelled) setEditsReady(true); });
    return () => { cancelled = true; };
  }, [result, file]);

  const chords: ChordSegment[] = editedChords ?? result?.chords ?? [];

  // Chords already in the song, most-used first, as one-click replacements.
  const quickPicks = useMemo(() => {
    const time = new Map<string, number>();
    for (const c of chords) {
      if (c.label !== "N") time.set(c.label, (time.get(c.label) ?? 0) + (c.end - c.start));
    }
    return Array.from(time.entries()).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([l]) => l);
  }, [chords]);

  const persist = (next: ChordSegment[] | null) => {
    setSaveFailed(!(editsKey && result && saveEdits(editsKey, result.duration, next)));
  };

  const commitEdit = (next: ChordSegment[]) => {
    // Deleting the only chord, or relabeling to the same thing, changes nothing
    // and shouldn't cost an undo step.
    if (JSON.stringify(next) === JSON.stringify(chords)) return;
    setUndoStack((s) => [...s, editedChords]);
    setRedoStack([]);
    setEditedChords(next);
    persist(next);
  };

  const applyLabel = (label: string) => {
    if (selectedIdx === null) return;
    const sel = chords[selectedIdx];
    const next = relabelChord(chords, selectedIdx, label);
    commitEdit(next);
    // Merging with a neighbor can shift indices; keep the same moment selected.
    const mid = (sel.start + sel.end) / 2;
    const idx = next.findIndex((c) => mid >= c.start && mid < c.end);
    setSelectedIdx(idx >= 0 ? idx : null);
  };

  // Insert at the playhead, snapped to the nearest beat since a pause by ear
  // lands a little late. Asks wavesurfer directly: currentTime lags a click
  // on the waveform by an event.
  const startInsert = () => {
    if (!result) return;
    const playhead = waveSurferRef.current?.getCurrentTime() ?? currentTime;
    const t = snapToBeat(playhead, result.beats);
    if (!chords.some((c) => t >= c.start && t < c.end)) return;
    setSelectedIdx(null);
    setInsertAt(t);
  };

  const applyInsert = (label: string) => {
    if (insertAt === null) return;
    const next = insertChord(chords, insertAt, label);
    commitEdit(next);
    const idx = next.findIndex((c) => insertAt >= c.start && insertAt < c.end);
    setInsertAt(null);
    setSelectedIdx(idx >= 0 ? idx : null);
  };

  // While inserting, the strip previews the result with a "?" block, so it's
  // clear which chord gets split (or wholly replaced) before anything changes.
  const insertPreview = (() => {
    if (insertAt === null || !result) return null;
    const host = chords.find((c) => insertAt >= c.start && insertAt < c.end);
    const after = insertChord(chords, insertAt, PENDING_LABEL);
    const index = after.findIndex((c) => c.label === PENDING_LABEL);
    if (!host || index < 0) return null;
    const pending = after[index];
    const name = (l: string) => (l === "N" ? "no chord" : formatChord(parseChord(transposeChord(l, semitones))));
    const note = after.length > chords.length
      ? `New chord ${formatTimeTenths(pending.start)}–${formatTimeTenths(pending.end)}. ` +
        `${name(host.label)} keeps ${formatTimeTenths(host.start)}–${formatTimeTenths(pending.start)}.`
      : `Replaces all of ${name(host.label)} (${formatTimeTenths(pending.start)}–${formatTimeTenths(pending.end)}). ` +
        `Use Beat ▶ to split it instead.`;
    const canGo = (dir: -1 | 1) => {
      const t = nudgeToBeat(insertAt, result.beats, dir, result.duration);
      return t !== null && chords.some((c) => t >= c.start && t < c.end);
    };
    return {
      chords: after,
      index,
      span: { start: pending.start, end: pending.end, label: host.label },
      note,
      canNudge: { earlier: canGo(-1), later: canGo(1) },
    };
  })();

  const nudgeInsert = (dir: -1 | 1) => {
    if (insertAt === null || !result) return;
    const t = nudgeToBeat(insertAt, result.beats, dir, result.duration);
    if (t !== null) setInsertAt(t);
  };

  // Strip indices refer to the preview while inserting, so a click there
  // just cancels the insert.
  const selectChord = (i: number) => {
    if (insertAt !== null) {
      setInsertAt(null);
      return;
    }
    setSelectedIdx(i);
  };

  const deleteSelected = () => {
    if (selectedIdx === null) return;
    commitEdit(deleteChord(chords, selectedIdx));
    setSelectedIdx(null);
  };

  const undoEdit = () => {
    if (undoStack.length === 0) return;
    const prev = undoStack[undoStack.length - 1];
    setUndoStack((s) => s.slice(0, -1));
    setRedoStack((s) => [...s, editedChords]);
    setEditedChords(prev);
    setSelectedIdx(null);
    persist(prev);
  };

  const redoEdit = () => {
    if (redoStack.length === 0) return;
    const next = redoStack[redoStack.length - 1];
    setRedoStack((s) => s.slice(0, -1));
    setUndoStack((s) => [...s, editedChords]);
    setEditedChords(next);
    setSelectedIdx(null);
    persist(next);
  };

  const revertEdits = () => {
    if (!window.confirm("Discard all your chord edits for this song?")) return;
    setEditedChords(null);
    setUndoStack([]);
    setRedoStack([]);
    setSelectedIdx(null);
    persist(null);
  };

  // Delete removes the selected chord, typing a chord name starts renaming it,
  // I or Insert inserts a chord at the playhead, Esc deselects, Ctrl/Cmd+Z
  // undoes, Ctrl+Shift+Z or Ctrl+Y redoes — except while typing in a field.
  // Held keys don't repeat edits. The listener reads the latest handlers
  // through a ref so it only re-binds when editing toggles.
  const onEditKey = (e: KeyboardEvent) => {
    const el = e.target as HTMLElement | null;
    if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")) return;
    const mod = e.ctrlKey || e.metaKey;
    const key = e.key.toLowerCase();
    if (e.key === "Delete" && selectedIdx !== null) {
      e.preventDefault();
      if (!e.repeat) deleteSelected();
    } else if (e.key === "Escape") {
      setSelectedIdx(null);
      setInsertAt(null);
    } else if (!mod && !e.altKey && (key === "i" || e.key === "Insert")) {
      e.preventDefault();
      if (!e.repeat) startInsert();
    } else if (mod && (key === "y" || (key === "z" && e.shiftKey))) {
      e.preventDefault();
      if (!e.repeat) redoEdit();
    } else if (mod && key === "z") {
      e.preventDefault();
      if (!e.repeat) undoEdit();
    } else if (!mod && !e.altKey && /^[a-gn]$/.test(key) && selectedIdx !== null) {
      // Focus moves before the key's character is inserted, so it lands in
      // the (select-all-on-focus) chord field and replaces its text.
      chordInputRef.current?.focus();
    }
  };
  const onEditKeyRef = useRef(onEditKey);
  onEditKeyRef.current = onEditKey;
  useEffect(() => {
    if (!editing) return;
    const onKey = (e: KeyboardEvent) => onEditKeyRef.current(e);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editing]);

  // ---- Wavesurfer setup --------------------------------------------------

  // The wave container is rendered conditionally on `result`, so we must
  // wait for both `file` AND `result` before creating the wavesurfer instance —
  // otherwise the container ref is null when the effect fires.
  useEffect(() => {
    if (!file || !result || !waveContainerRef.current) return;

    if (waveSurferRef.current) {
      waveSurferRef.current.destroy();
      waveSurferRef.current = null;
    }

    const ws = WaveSurfer.create({
      container: waveContainerRef.current,
      waveColor: "#3a4255",
      progressColor: "#7aa2f7",
      cursorColor: "#a3c5ff",
      height: 80,
      barWidth: 2,
      barGap: 1,
    });

    // Apply the current slider volume immediately so newly-loaded audio
    // starts at the right level rather than wavesurfer's default.
    try { ws.setVolume(volume); } catch { /* no-op */ }

    ws.loadBlob(file).catch((err) => {
      console.error("wavesurfer load failed:", err);
      setError(`Audio load failed: ${err?.message ?? err}`);
    });
    ws.on("audioprocess", () => setCurrentTime(ws.getCurrentTime()));
    ws.on("seeking", () => setCurrentTime(ws.getCurrentTime()));
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));

    waveSurferRef.current = ws;

    return () => {
      ws.destroy();
      waveSurferRef.current = null;
    };
  }, [file, result]);

  const togglePlay = () => waveSurferRef.current?.playPause();

  // Space toggles play/pause anywhere except while typing in a field. Keyup is
  // swallowed too, or a focused button (Play, Insert chord…) would also click.
  useEffect(() => {
    const isSpace = (e: KeyboardEvent) => {
      if (e.code !== "Space" && e.key !== " ") return false;
      const el = e.target as HTMLElement | null;
      const typing = el && (
        (el.tagName === "INPUT" && !["checkbox", "radio", "range", "button"].includes((el as HTMLInputElement).type)) ||
        el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable
      );
      return !typing && !e.ctrlKey && !e.metaKey && !e.altKey;
    };
    const onDown = (e: KeyboardEvent) => {
      if (!isSpace(e) || !waveSurferRef.current) return;
      e.preventDefault();
      if (!e.repeat) waveSurferRef.current.playPause();
    };
    const onUp = (e: KeyboardEvent) => {
      if (isSpace(e) && waveSurferRef.current) e.preventDefault();
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, []);
  const seek = (time: number) => {
    if (waveSurferRef.current && result) {
      waveSurferRef.current.seekTo(time / result.duration);
    }
  };

  // ---- Export ------------------------------------------------------------

  const onExport = () => {
    if (!result || !file) return;
    const content = exportChordPro({
      filename: file.name,
      key: result.key,
      bpm: result.bpm,
      duration: result.duration,
      chords,
      beats: result.beats,
      lyrics: result.lyrics ?? null,
      semitones,
    });
    const base = file.name.replace(/\.[^.]+$/, "");
    const suffix = semitones === 0 ? "" : `_${semitones > 0 ? "+" : ""}${semitones}`;
    downloadText(content, `${base}${suffix}.chordpro.txt`);
  };

  const onExportTab = () => {
    if (!result || !file) return;
    const content = exportTab({
      filename: file.name,
      key: result.key,
      bpm: result.bpm,
      duration: result.duration,
      chords,
      lyrics: result.lyrics ?? null,
      sections: result.sections ?? null,
      capo: capoSuggestion,
      semitones,
    });
    const base = file.name.replace(/\.[^.]+$/, "");
    const suffix = semitones === 0 ? "" : `_${semitones > 0 ? "+" : ""}${semitones}`;
    downloadText(content, `${base}${suffix}.tab.txt`);
  };

  // ---- Capo suggestion ---------------------------------------------------

  // If the song is in a "hard" guitar key (Eb, Ab, Db, etc.), suggest a capo
  // that puts it in an easy guitar key (C, G, D, A, E).
  const capoSuggestion = (() => {
    if (!result) return null;
    const easyKeys = new Set(["C", "G", "D", "A", "E"]);
    const [root, mode] = result.key.split(" ");
    const SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const rootIdx = SHARPS.indexOf(root.replace("b", ""));
    if (rootIdx < 0 || easyKeys.has(root)) return null;
    for (let capo = 1; capo <= 7; capo++) {
      // A capo on fret N means we're playing chord shapes a perfect Nth lower
      // than the sounding pitch. So the shape key = sounding key - capo.
      const shapeIdx = (rootIdx - capo + 12) % 12;
      const shapeRoot = SHARPS[shapeIdx];
      if (easyKeys.has(shapeRoot)) {
        return { capo, shapeKey: `${shapeRoot} ${mode}` };
      }
    }
    return null;
  })();

  // ---- Render ------------------------------------------------------------

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px" }}>
      <h1 style={{ fontSize: 28, marginBottom: 4 }}>Chord Detector</h1>
      <p style={{ color: "var(--muted)", marginTop: 0, marginBottom: 24 }}>
        Chord detection, lyric transcription, and section analysis. Stems run
        locally; chord ID and lyrics can optionally use Gemini.
      </p>

      {/* Upload */}
      <div
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        style={{
          border: "2px dashed var(--border)",
          borderRadius: 8,
          padding: 24,
          textAlign: "center",
          marginBottom: 24,
          background: "var(--panel)",
        }}
      >
        <input
          type="file"
          accept=".mp3,.wav,.flac,.m4a,.ogg,.aac"
          onChange={onFileChange}
          style={{ display: "none" }}
          id="file-input"
        />
        <label htmlFor="file-input" style={{ cursor: "pointer" }}>
          <div style={{ fontSize: 16, marginBottom: 8 }}>
            {file ? file.name : "Drop an audio file or click to choose"}
          </div>
          <div style={{ fontSize: 12, color: "var(--muted)" }}>
            mp3, wav, flac, m4a, ogg, aac
          </div>
        </label>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 24,
            marginTop: 14,
            flexWrap: "wrap",
            fontSize: 13,
            color: "var(--muted)",
          }}
        >
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={separateStems}
              onChange={(e) => setSeparateStems(e.target.checked)}
              disabled={loading}
            />
            Separate stems first (slower, much better on full-band mixes)
          </label>
          <label
            title={separateStems ? "" : "Requires stem separation"}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              cursor: separateStems && !loading ? "pointer" : "not-allowed",
              opacity: separateStems ? 1 : 0.5,
            }}
          >
            <input
              type="checkbox"
              checked={transcribeLyrics}
              onChange={(e) => setTranscribeLyrics(e.target.checked)}
              disabled={loading || !separateStems}
            />
            Transcribe lyrics (requires stems, adds ~1 min)
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            Chord vocabulary:
            <select
              value={vocabulary}
              onChange={(e) => setVocabulary(e.target.value as typeof vocabulary)}
              disabled={loading}
              style={{ background: "var(--panel)", color: "var(--text)", border: "1px solid var(--border)", padding: "2px 6px", borderRadius: 4 }}
            >
              <option value="triads">Triads (maj/min)</option>
              <option value="triads_7ths">Triads + 7ths</option>
              <option value="extended">Extended (sus, dim, aug, 7ths)</option>
            </select>
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            Chord detection:
            <select
              value={chordProvider}
              onChange={(e) => setChordProvider(e.target.value as typeof chordProvider)}
              disabled={loading}
              style={{ background: "var(--panel)", color: "var(--text)", border: "1px solid var(--border)", padding: "2px 6px", borderRadius: 4 }}
            >
              <option value="hybrid">Hybrid (Gemini + local)</option>
              <option value="local">Local only</option>
            </select>
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            Recognizer:
            <select
              value={chordRecognizer}
              onChange={(e) => setChordRecognizer(e.target.value as typeof chordRecognizer)}
              disabled={loading}
              style={{ background: "var(--panel)", color: "var(--text)", border: "1px solid var(--border)", padding: "2px 6px", borderRadius: 4 }}
            >
              <option value="local">Pearson + Viterbi</option>
              <option value="btc">BTC transformer (experimental)</option>
            </select>
          </label>
          {chordRecognizer === "btc" && chordProvider === "hybrid" && (
            <span style={{ fontSize: 12, color: "var(--muted, #999)", fontStyle: "italic" }}>
              Note: BTC ignores the Gemini chord set — hybrid mode has no effect.
            </span>
          )}
          {(chordProvider === "hybrid" || (transcribeLyrics && lyricProvider === "gemini")) && (
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              Gemini model:
              <select
                value={geminiModel}
                onChange={(e) => setGeminiModel(e.target.value as typeof geminiModel)}
                disabled={loading}
                style={{ background: "var(--panel)", color: "var(--text)", border: "1px solid var(--border)", padding: "2px 6px", borderRadius: 4 }}
              >
                <option value="gemini-3.5-flash">3.5 Flash (fast, free)</option>
                <option value="gemini-3.6-flash">3.6 Flash (newer, free)</option>
                <option value="gemini-3.1-pro-preview">3.1 Pro Preview (slower, better)</option>
              </select>
            </label>
          )}
          {transcribeLyrics && lyricProvider === "gemini" && (
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              Lyric model:
              <select
                value={geminiLyricModel}
                onChange={(e) => setGeminiLyricModel(e.target.value as typeof geminiLyricModel)}
                disabled={loading}
                style={{ background: "var(--panel)", color: "var(--text)", border: "1px solid var(--border)", padding: "2px 6px", borderRadius: 4 }}
              >
                <option value="">Same as Gemini model</option>
                <option value="gemini-3.5-flash">3.5 Flash</option>
                <option value="gemini-3.6-flash">3.6 Flash</option>
                <option value="gemini-3.8-flash">3.8 Flash (lyrics only)</option>
                <option value="gemini-3.1-pro-preview">3.1 Pro Preview</option>
              </select>
            </label>
          )}
        </div>
        {transcribeLyrics && (
          <>
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                gap: 16,
                marginTop: 10,
                flexWrap: "wrap",
                fontSize: 13,
                color: "var(--muted)",
              }}
            >
              <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                Provider:
                <select
                  value={lyricProvider}
                  onChange={(e) => setLyricProvider(e.target.value as typeof lyricProvider)}
                  disabled={loading}
                  style={{ background: "var(--panel)", color: "var(--text)", border: "1px solid var(--border)", padding: "2px 6px", borderRadius: 4 }}
                >
                  <option value="gemini">Gemini (cloud, best on music)</option>
                  <option value="whisper">Whisper (local)</option>
                </select>
              </label>
              {lyricProvider === "whisper" && (
                <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  Whisper model:
                  <select
                    value={whisperModel}
                    onChange={(e) => setWhisperModel(e.target.value as typeof whisperModel)}
                    disabled={loading}
                    style={{ background: "var(--panel)", color: "var(--text)", border: "1px solid var(--border)", padding: "2px 6px", borderRadius: 4 }}
                  >
                    <option value="small">small (fast, ~480MB)</option>
                    <option value="medium">medium (better, ~1.5GB, ~3× slower)</option>
                    <option value="large-v3">large-v3 (best, ~3GB, ~5× slower)</option>
                  </select>
                </label>
              )}
              <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                Hint:
                <input
                  type="text"
                  value={lyricHint}
                  onChange={(e) => setLyricHint(e.target.value)}
                  disabled={loading}
                  placeholder="e.g. band name, genre"
                  style={{
                    background: "var(--panel)",
                    color: "var(--text)",
                    border: "1px solid var(--border)",
                    padding: "2px 6px",
                    borderRadius: 4,
                    width: 220,
                  }}
                />
              </label>
            </div>
            <div
              style={{
                marginTop: 6,
                fontSize: 11,
                color: "var(--muted)",
                opacity: 0.75,
                textAlign: "center",
              }}
            >
              {lyricProvider === "gemini"
                ? "Gemini sends the vocals stem to Google. Falls back to local Whisper on error."
                : "Whisper runs locally — audio stays on this machine."}
            </div>
          </>
        )}
      </div>

      {loading && (
        <div style={{ padding: 16, color: "var(--muted)" }}>
          <div>
            {transcribeLyrics
              ? "Analyzing with stems + lyric transcription… (2–4 min; first run also downloads the Whisper model)"
              : separateStems
              ? "Analyzing with stem separation… (1–3 min; first run also downloads the demucs model)"
              : "Analyzing… (this takes 10–30s for a typical song)"}
          </div>
          <div
            style={{
              marginTop: 10,
              display: "flex",
              alignItems: "center",
              gap: 12,
            }}
          >
            <div
              style={{
                flex: 1,
                height: 6,
                background: "var(--border)",
                borderRadius: 3,
                overflow: "hidden",
              }}
            >
              <div
                data-testid="progress-fill"
                style={{
                  width: `${progress}%`,
                  height: "100%",
                  background: "var(--accent)",
                  transition: "width 300ms ease-out",
                }}
              />
            </div>
            <span
              style={{
                fontFamily: "monospace",
                fontSize: 12,
                color: "var(--muted)",
                minWidth: 60,
                textAlign: "right",
              }}
            >
              {formatTime(elapsedSec)} · {progress}%
            </span>
          </div>
        </div>
      )}

      {error && (
        <div
          style={{
            padding: 16,
            background: "#3a1f23",
            border: "1px solid #6b3033",
            borderRadius: 8,
            color: "#ffb3b3",
            marginBottom: 16,
          }}
        >
          {error}
        </div>
      )}

      {result && (
        <>
          {/* Meta row */}
          <div
            style={{
              display: "flex",
              gap: 24,
              padding: "12px 16px",
              background: "var(--panel)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              marginBottom: 16,
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            <div>
              <span style={{ color: "var(--muted)", fontSize: 12 }}>KEY </span>
              <strong>{transposeKey(result.key, semitones)}</strong>
            </div>
            <div>
              <span style={{ color: "var(--muted)", fontSize: 12 }}>BPM </span>
              <strong>{Math.round(result.bpm)}</strong>
            </div>
            <div>
              <span style={{ color: "var(--muted)", fontSize: 12 }}>CHORDS </span>
              <strong>{chords.filter((c) => c.label !== "N").length}</strong>
              {editedChords && (
                <span style={{ color: "var(--accent)", fontSize: 12, marginLeft: 6 }}>✎ edited</span>
              )}
            </div>
            {processingTimeSec !== null && (
              <div>
                <span style={{ color: "var(--muted)", fontSize: 12 }}>PROCESSED IN </span>
                <strong style={{ fontFamily: "ui-monospace, monospace" }}>
                  {formatTime(Math.round(processingTimeSec))}
                </strong>
              </div>
            )}
            {result.lyric_provider_used === "whisper-fallback" && (
              <div style={{ color: "var(--muted)", fontSize: 12, fontStyle: "italic" }}>
                ⚠ Gemini lyrics failed — used Whisper small as fallback
              </div>
            )}
            {result.chord_provider_used === "local-fallback" && (
              <div style={{ color: "var(--muted)", fontSize: 12, fontStyle: "italic" }}>
                ⚠ Gemini chord ID failed — used full local template set as fallback
              </div>
            )}
            {result.gemini_chord_set && result.gemini_chord_set.length > 0 && (
              <div style={{ color: "var(--accent)", fontSize: 12 }}>
                🎯 Chord set from Gemini: {result.gemini_chord_set.join(" ")}
              </div>
            )}
            {capoSuggestion && semitones === 0 && (
              <div style={{ color: "var(--accent)", fontSize: 13 }}>
                💡 Capo {capoSuggestion.capo} → play in {capoSuggestion.shapeKey}
              </div>
            )}

            {/* Transpose */}
            <div style={{ marginLeft: "auto", display: "flex", gap: 6, alignItems: "center" }}>
              <span style={{ color: "var(--muted)", fontSize: 12 }}>TRANSPOSE</span>
              <button onClick={() => setSemitones((s) => s - 1)}>−</button>
              <span
                style={{
                  minWidth: 32,
                  textAlign: "center",
                  fontFamily: "ui-monospace, monospace",
                }}
              >
                {semitones > 0 ? `+${semitones}` : semitones}
              </span>
              <button onClick={() => setSemitones((s) => s + 1)}>+</button>
              <button onClick={() => setSemitones(0)} disabled={semitones === 0}>
                Reset
              </button>
            </div>
          </div>

          {/* Waveform + transport */}
          <div
            style={{
              padding: 16,
              background: "var(--panel)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              marginBottom: 16,
            }}
          >
            <div ref={waveContainerRef} style={{ marginBottom: 12 }} />
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <button onClick={togglePlay} title="Play/Pause (Space)" style={{ minWidth: 80 }}>
                {isPlaying ? "Pause" : "Play"}
              </button>
              <span style={{ color: "var(--muted)", fontSize: 13, fontFamily: "monospace" }}>
                {formatTime(currentTime)} / {formatTime(result.duration)}
              </span>
              <label
                title="Output volume"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  marginLeft: 12,
                  color: "var(--muted)",
                  fontSize: 12,
                }}
              >
                🔊
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={volume}
                  onChange={(e) => setVolume(parseFloat(e.target.value))}
                  style={{ width: 100 }}
                />
                <span style={{ width: 32, textAlign: "right", fontFamily: "monospace" }}>
                  {Math.round(volume * 100)}
                </span>
              </label>
              <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
                <button onClick={onExport}>Export ChordPro</button>
                <button onClick={onExportTab}>Export Tab</button>
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
            <button
              onClick={() => { setEditing((e) => !e); setSelectedIdx(null); setInsertAt(null); }}
              disabled={!editsReady}
              style={editing ? { borderColor: "var(--accent)", color: "var(--accent-bright)" } : undefined}
            >
              {editing ? "Done editing" : "Edit chords"}
            </button>
            {editing && (
              <>
                <button
                  onClick={startInsert}
                  title="Insert a chord change at the playhead, snapped to the nearest beat (I)"
                >
                  Insert chord
                </button>
                <button onClick={undoEdit} disabled={undoStack.length === 0} title="Undo (Ctrl+Z)">
                  Undo
                </button>
                <button onClick={redoEdit} disabled={redoStack.length === 0} title="Redo (Ctrl+Shift+Z)">
                  Redo
                </button>
              </>
            )}
            {editedChords && (
              <button onClick={revertEdits}>Revert to detected</button>
            )}
            <span style={{ color: saveFailed ? "#e06c75" : "var(--muted)", fontSize: 12 }}>
              {saveFailed
                ? "Couldn't save edits in this browser. They'll be lost when you reload."
                : editing
                  ? "Click a chord to rename or delete it. To add a missed change, put the playhead there and press Insert chord."
                  : editedChords
                    ? "Showing your edits. They're saved in this browser."
                    : ""}
            </span>
          </div>

          <ChordTimeline
            chords={insertPreview?.chords ?? chords}
            pendingIndex={insertPreview?.index ?? null}
            duration={result.duration}
            currentTime={currentTime}
            semitones={semitones}
            onSeek={seek}
            editing={editing}
            selectedIndex={selectedIdx}
            onSelect={selectChord}
          />

          {editing && insertPreview && (
            <ChordEditBar
              mode="insert"
              chord={insertPreview.span}
              insertNote={insertPreview.note}
              onNudge={nudgeInsert}
              canNudge={insertPreview.canNudge}
              semitones={semitones}
              quickPicks={quickPicks}
              canDelete={false}
              inputRef={chordInputRef}
              onApply={applyInsert}
              onDelete={() => {}}
              onClose={() => setInsertAt(null)}
            />
          )}

          {editing && !insertPreview && selectedIdx !== null && chords[selectedIdx] && (
            <ChordEditBar
              chord={chords[selectedIdx]}
              semitones={semitones}
              quickPicks={quickPicks}
              canDelete={chords.length > 1}
              inputRef={chordInputRef}
              onApply={applyLabel}
              onDelete={deleteSelected}
              onClose={() => setSelectedIdx(null)}
            />
          )}
        </>
      )}
    </main>
  );
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s - m * 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}

function formatTimeTenths(s: number): string {
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}
