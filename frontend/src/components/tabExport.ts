import { ChordSegment, LyricWord, Section } from "./types";
import { parseChord, formatChord, transposeChord, transposeKey } from "./chord";

/**
 * Ultimate Guitar–style tab export: monospaced text with chord names
 * positioned above the lyric word they fall on, plus section markers
 * like [Verse 1] / [Chorus].
 *
 * Looks correct only in a monospaced editor — same constraint as UG.
 */
export function exportTab(opts: {
  filename: string;
  key: string;
  bpm: number;
  duration: number;
  chords: ChordSegment[];
  lyrics?: LyricWord[] | null;
  sections?: Section[] | null;
  capo?: { capo: number; shapeKey: string } | null;
  semitones: number;
}): string {
  const { filename, key, bpm, duration, chords, lyrics, sections, capo, semitones } = opts;
  const out: string[] = [];

  // ---- Header ----------------------------------------------------------
  out.push(stripExt(filename));
  const headerBits = [
    "Tuning: E A D G B E",
    `Key: ${transposeKey(key, semitones)}`,
    `Capo: ${capo && semitones === 0 ? `${capo.capo} (play in ${capo.shapeKey})` : "No capo"}`,
    `Tempo: ${Math.round(bpm)}`,
  ];
  out.push(headerBits.join("   ·   "));

  // ---- Section walk ----------------------------------------------------
  const safeSections =
    sections && sections.length > 0
      ? sections
      : [{ start: 0, end: duration, label: "Song" }];

  for (const sec of safeSections) {
    out.push("");
    out.push(`[${sec.label}]`);

    const wordsInSection = (lyrics ?? []).filter(
      (w) => w.start >= sec.start && w.start < sec.end,
    );

    if (wordsInSection.length === 0) {
      // Instrumental section: list the chords used, in order, deduped on
      // adjacent repeats. Then collapse exact-period repetition into
      // `pattern  (xN)` form — UG's notation for vamp-style intros/outros.
      const chordsInSection = chords.filter(
        (c) => c.start < sec.end && c.end > sec.start && c.label !== "N",
      );
      const seq: string[] = [];
      let prev = "";
      for (const c of chordsInSection) {
        const label = display(c.label, semitones);
        if (label !== prev) {
          seq.push(label);
          prev = label;
        }
      }
      if (seq.length) {
        const { pattern, reps, remainder } = detectRepeatingPattern(seq);
        if (reps >= 2) {
          const head = `${pattern.join("  ")}  (x${reps})`;
          out.push(
            remainder.length > 0
              ? `${head}  ${remainder.join("  ")}`
              : head,
          );
        } else {
          out.push(seq.join("  "));
        }
      }
      continue;
    }

    // Group section words into lines: break on word gaps > 1.2s, OR when the
    // running lyric line would exceed ~80 characters.
    const lines = groupWordsIntoLines(wordsInSection);

    // Chord changes inside this section, sorted by start, excluding N.
    const sectionChordChanges = chords
      .filter((c) => c.start >= sec.start && c.start < sec.end && c.label !== "N")
      .sort((a, b) => a.start - b.start);

    let changeCursor = 0;

    for (const line of lines) {
      let chordLine = "";
      let lyricLine = "";

      const firstWordStart = line.words[0].start;

      // UG convention: every lyric line gets a chord label above its first
      // word, showing whichever chord is currently sounding. This applies
      // even when no chord *change* happens between lines — the leading
      // chord is just the snapshot of "what's playing on the downbeat of
      // this phrase." Without this, lines with no mid-line chord changes
      // come out as bare lyric strings.
      const leadingForLine = chordAt(firstWordStart, chords);
      if (leadingForLine) {
        chordLine = display(leadingForLine, semitones);
      }

      // Advance the cursor past any chord changes that happened on or before
      // this line's first word — they're already represented by the leading
      // chord we just emitted, so we don't want to double up.
      while (
        changeCursor < sectionChordChanges.length &&
        sectionChordChanges[changeCursor].start <= firstWordStart
      ) {
        changeCursor++;
      }

      for (const w of line.words) {
        // Place any chord changes whose time is at-or-before this word's
        // start at the current lyric cursor (= current length of lyricLine).
        while (
          changeCursor < sectionChordChanges.length &&
          sectionChordChanges[changeCursor].start <= w.start
        ) {
          const label = display(sectionChordChanges[changeCursor].label, semitones);
          chordLine = padTo(chordLine, lyricLine.length) + label;
          changeCursor++;
        }
        lyricLine += escapeText(w.word) + " ";
      }

      // Emit chord line (if non-empty) then lyric line.
      const trimmedChord = chordLine.replace(/\s+$/, "");
      if (trimmedChord.length > 0) out.push(trimmedChord);
      out.push(lyricLine.replace(/\s+$/, ""));
    }
  }

  return out.join("\n") + "\n";
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface Line {
  words: LyricWord[];
}

function groupWordsIntoLines(words: LyricWord[]): Line[] {
  // Phrase-aware line breaks. Three triggers, in order of precedence:
  //   1. Capitalized word — Gemini sentence-cases the start of each phrase,
  //      so a multi-letter capitalized word reliably signals a new line.
  //      Excludes single-letter caps (band/proper-noun fragments like "B")
  //      and the lone "I" pronoun + contractions.
  //   2. Word gap > 0.6s — natural breath/pause.
  //   3. Hard cap of 12 words per line — safety net when Gemini neither
  //      capitalizes nor leaves a gap.
  if (words.length === 0) return [];

  const MAX_WORDS_PER_LINE = 12;
  const GAP_THRESHOLD_SEC = 0.6;

  const isPhraseStart = (w: string): boolean => {
    if (!w || w.length < 2) return false; // skip "B", "A" as standalone
    const first = w[0];
    if (first < "A" || first > "Z") return false;
    // "I", "I'm", "I'll", "I've", "I'd" — straight + curly apostrophes
    if (/^I['‘’]/.test(w)) return false;
    return true;
  };

  const lines: Line[] = [];
  let current: LyricWord[] = [];

  const flush = () => {
    if (current.length > 0) {
      lines.push({ words: current });
      current = [];
    }
  };

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const prevEnd = current.length > 0 ? current[current.length - 1].end : null;
    const gap = prevEnd !== null ? w.start - prevEnd : 0;

    const shouldBreak =
      current.length > 0 &&
      (
        isPhraseStart(w.word) ||
        gap > GAP_THRESHOLD_SEC ||
        current.length >= MAX_WORDS_PER_LINE
      );

    if (shouldBreak) flush();
    current.push(w);
  }
  flush();
  return lines;
}

function chordAt(t: number, chords: ChordSegment[]): string | null {
  for (const c of chords) {
    if (t >= c.start && t < c.end) return c.label;
  }
  if (chords.length > 0 && t >= chords[chords.length - 1].end) {
    return chords[chords.length - 1].label;
  }
  return null;
}

function display(label: string, semitones: number): string {
  if (!label || label === "N") return "";
  return formatChord(parseChord(transposeChord(label, semitones)));
}

function padTo(s: string, len: number): string {
  if (s.length >= len) return s + " "; // ensure at least one space between adjacent chords
  return s + " ".repeat(len - s.length);
}

function escapeText(s: string): string {
  // Strip any character that would visually break the chord-over-word
  // alignment trick or be confused for a directive.
  return s.replace(/[\r\n\t]/g, "");
}

function stripExt(filename: string): string {
  return filename.replace(/\.[^.]+$/, "");
}

/**
 * Find a leading repeating pattern in `seq`. Returns the pattern, the number
 * of repetitions (>= 2), and any trailing remainder that didn't match. If no
 * pattern repeats at least twice, returns the original sequence with reps=1
 * and an empty remainder.
 *
 * Picks the period that maximizes covered length (reps × period); ties go
 * to the shorter period. Used to collapse instrumental vamps like
 * `G D Am Em G D Am Em G D Am Em G E G E Gm` into UG-style
 * `G D Am Em (x3) G E G E Gm`.
 */
function detectRepeatingPattern(
  seq: string[],
): { pattern: string[]; reps: number; remainder: string[] } {
  const n = seq.length;
  if (n < 4) return { pattern: seq, reps: 1, remainder: [] };

  let bestP = 0;
  let bestK = 1;
  let bestCovered = 0;

  for (let p = 1; p <= Math.floor(n / 2); p++) {
    // How many consecutive copies of seq[0..p] does the prefix contain?
    let k = 1;
    while ((k + 1) * p <= n) {
      let match = true;
      for (let j = 0; j < p; j++) {
        if (seq[k * p + j] !== seq[j]) {
          match = false;
          break;
        }
      }
      if (!match) break;
      k++;
    }
    if (k >= 2) {
      const covered = k * p;
      // Prefer longer total coverage; on a tie, prefer the shorter period.
      if (covered > bestCovered || (covered === bestCovered && p < bestP)) {
        bestP = p;
        bestK = k;
        bestCovered = covered;
      }
    }
  }

  if (bestCovered === 0) return { pattern: seq, reps: 1, remainder: [] };
  return {
    pattern: seq.slice(0, bestP),
    reps: bestK,
    remainder: seq.slice(bestK * bestP),
  };
}
