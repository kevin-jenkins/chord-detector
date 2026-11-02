import { ChordSegment, LyricWord } from "./types";
import { parseChord, formatChord, transposeChord, transposeKey } from "./chord";

/**
 * Real ChordPro export with bar alignment.
 *
 * - Headers: {title}, {key}, {tempo}, {time:4/4}, optional {comment} for transpose.
 * - Bars are derived from the detected beats[] in groups of 4 (4/4 assumed;
 *   the first beat is treated as the downbeat — wrong ~30% of the time but
 *   close enough for a usable chart).
 * - Each bar is classified LYRIC (contains a sung word) or INSTRUMENTAL.
 *     LYRIC bars render inline:    [D]word1 [A]word2 [Em]word3
 *     INSTRUMENTAL bars render:    | D . . . | A . . . | Em . . . | G . . . |
 *   Up to 4 instrumental bars are joined onto a single line.
 * - "N" chord segments (silence / no chord detected) are rendered as `.`.
 */
export function exportChordPro(opts: {
  filename: string;
  key: string;
  bpm: number;
  duration: number;
  chords: ChordSegment[];
  beats: number[];
  lyrics?: LyricWord[] | null;
  semitones: number;
}): string {
  const { filename, key, bpm, duration, chords, beats, lyrics, semitones } = opts;
  const hasLyrics = !!lyrics && lyrics.length > 0;

  const lines: string[] = [];
  lines.push(`{title: ${filename}}`);
  lines.push(`{key: ${transposeKey(key, semitones)}}`);
  lines.push(`{tempo: ${Math.round(bpm)}}`);
  lines.push(`{time: 4/4}`);
  if (semitones !== 0) {
    lines.push(`{comment: transposed ${semitones > 0 ? "+" : ""}${semitones} semitones}`);
  }
  lines.push("");

  // Fallback: no beats detected → emit the legacy chord-timeline format so
  // we still produce something useful.
  if (beats.length < 2) {
    for (const c of chords) {
      if (c.label === "N") continue;
      const label = display(c.label, semitones);
      lines.push(`[${formatTime(c.start)}] ${label.padEnd(8)} (${(c.end - c.start).toFixed(1)}s)`);
    }
    return lines.join("\n");
  }

  const bars = buildBars(beats, duration);
  const lyricBuckets = hasLyrics ? bucketLyrics(lyrics!, bars) : null;

  // Group consecutive INSTRUMENTAL bars and emit them 4-per-line.
  let instrumentalRun: string[] = [];
  const flushInstrumental = () => {
    while (instrumentalRun.length > 0) {
      const chunk = instrumentalRun.splice(0, 4);
      lines.push(chunk.join(" ") + " |");
    }
  };

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const bucket = lyricBuckets ? lyricBuckets[i] : [];
    const isLyric = bucket.length > 0;

    if (isLyric) {
      flushInstrumental();
      lines.push(renderLyricBar(bar, bucket, chords, semitones));
    } else {
      instrumentalRun.push(renderInstrumentalBar(bar, chords, semitones));
    }
  }
  flushInstrumental();

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Bar construction
// ---------------------------------------------------------------------------

interface Bar {
  start: number;       // seconds
  end: number;         // seconds
  beatTimes: number[]; // up to 4 beat timestamps inside this bar
}

function buildBars(beats: number[], duration: number): Bar[] {
  const bars: Bar[] = [];
  for (let i = 0; i < beats.length; i += 4) {
    const beatTimes = beats.slice(i, i + 4);
    const start = beatTimes[0];
    // End at the next bar's downbeat if it exists; otherwise extrapolate
    // using the average beat interval, capped by duration.
    let end: number;
    if (i + 4 < beats.length) {
      end = beats[i + 4];
    } else if (beatTimes.length >= 2) {
      const avgBeat = (beatTimes[beatTimes.length - 1] - beatTimes[0]) / (beatTimes.length - 1);
      end = Math.min(duration, beatTimes[beatTimes.length - 1] + avgBeat);
    } else {
      end = duration;
    }
    bars.push({ start, end, beatTimes });
  }
  return bars;
}

function bucketLyrics(lyrics: LyricWord[], bars: Bar[]): LyricWord[][] {
  const buckets: LyricWord[][] = bars.map(() => []);
  // Lyrics sorted by start; bars sorted by start. Two-pointer walk.
  let b = 0;
  for (const w of lyrics) {
    while (b < bars.length - 1 && w.start >= bars[b].end) b++;
    if (w.start >= bars[b].start && w.start < bars[b].end) {
      buckets[b].push(w);
    }
  }
  return buckets;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderLyricBar(
  bar: Bar,
  words: LyricWord[],
  chords: ChordSegment[],
  semitones: number,
): string {
  // Chord changes whose START falls inside this bar (the chord active at
  // bar.start is handled separately as the leading [chord]).
  const changes = chords.filter(
    (c) => c.start > bar.start && c.start < bar.end && c.label !== "N",
  );

  let out = "";
  const leading = chordAt(bar.start, chords);
  if (leading && leading !== "N") {
    out += `[${display(leading, semitones)}]`;
  }

  let cursor = 0;
  for (const w of words) {
    while (cursor < changes.length && changes[cursor].start <= w.start) {
      out += `[${display(changes[cursor].label, semitones)}]`;
      cursor++;
    }
    out += escapeChordPro(w.word) + " ";
  }
  // Any remaining chord changes after the last word in this bar — still
  // emit them so the chart stays accurate.
  while (cursor < changes.length) {
    out += `[${display(changes[cursor].label, semitones)}]`;
    cursor++;
  }
  return out.trimEnd();
}

function renderInstrumentalBar(bar: Bar, chords: ChordSegment[], semitones: number): string {
  // Pad to exactly 4 beat slots so bars line up visually.
  const slots: string[] = [];
  let prev: string | null = null;
  for (let i = 0; i < 4; i++) {
    const t = bar.beatTimes[i] ?? bar.start + ((bar.end - bar.start) * i) / 4;
    const ch = chordAt(t, chords);
    if (!ch || ch === "N") {
      slots.push(".");
      prev = null;
    } else if (ch === prev) {
      slots.push(".");
    } else {
      slots.push(display(ch, semitones));
      prev = ch;
    }
  }
  // Width align so columns roughly line up between bars.
  const padded = slots.map((s) => s.padEnd(4, " "));
  return `| ${padded.join(" ")}`;
}

function chordAt(t: number, chords: ChordSegment[]): string | null {
  // Linear scan is fine — chord arrays are typically O(100) segments.
  // Returns the label of the segment that covers `t`, or null.
  for (const c of chords) {
    if (t >= c.start && t < c.end) return c.label;
  }
  // Past the last segment end: assume the last chord still sustains.
  if (chords.length > 0 && t >= chords[chords.length - 1].end) {
    return chords[chords.length - 1].label;
  }
  return null;
}

function display(label: string, semitones: number): string {
  if (!label || label === "N") return ".";
  return formatChord(parseChord(transposeChord(label, semitones)));
}

function escapeChordPro(s: string): string {
  // ChordPro uses [ ] for chords and { } for directives. Real lyrics
  // basically never contain these, but strip them defensively.
  return s.replace(/[\[\]{}]/g, "");
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

export function downloadText(content: string, filename: string) {
  const blob = new Blob([content], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
