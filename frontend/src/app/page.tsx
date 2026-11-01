"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import ChordTimeline from "@/components/ChordTimeline";
import { AnalysisResult } from "@/components/types";
import { transposeKey } from "@/components/chord";
import { exportChordPro, downloadText } from "@/components/export";
import { exportTab } from "@/components/tabExport";

const API = "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
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
  // Model ids current as of Nov 2026. gemini-2.5-pro was deprecated 2026-10-16;
  // gemini-3.1-pro-preview is the current Pro API id. Keep in sync with
  // _GEMINI_MODELS in backend/main.py.
  const [geminiModel, setGeminiModel] = useState<"gemini-3.5-flash" | "gemini-3.6-flash" | "gemini-3.1-pro-preview">("gemini-3.1-pro-preview");
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
      analyze(f, separateStems, vocabulary, transcribeLyrics, whisperModel, lyricHint, lyricProvider, chordProvider, geminiModel, chordRecognizer);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) {
      setFile(f);
      analyze(f, separateStems, vocabulary, transcribeLyrics, whisperModel, lyricHint, lyricProvider, chordProvider, geminiModel, chordRecognizer);
    }
  };

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
      chords: result.chords,
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
      chords: result.chords,
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
              <strong>{result.chords.filter((c) => c.label !== "N").length}</strong>
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
              <button onClick={togglePlay} style={{ minWidth: 80 }}>
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

          <ChordTimeline
            chords={result.chords}
            duration={result.duration}
            currentTime={currentTime}
            semitones={semitones}
            onSeek={seek}
          />
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
