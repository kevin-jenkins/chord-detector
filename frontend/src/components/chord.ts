/**
 * Chord label utilities.
 *
 * The backend emits labels in shorthand form:
 *   "C", "Cm", "C7", "Cmaj7", "Cm7", "Cdim", "Caug", "Csus2", "Csus4",
 *   slash chords like "G/B", and "N" for no-chord.
 *
 * Older Chordino-style colon labels ("C:maj7", "C:min") are also tolerated
 * so this stays compatible if we ever swap recognizers again.
 */

const SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLATS  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

export interface ParsedChord {
  root: string | null;   // null for "N"
  quality: string;       // "", "m", "7", "maj7", "m7", "dim", "aug", "sus4", "sus2"
  bass: string | null;   // for slash chords
}

function pitchToIndex(p: string): number {
  // Accept "C", "C#", "Db", etc.
  const norm = p.replace("b", "♭").replace("#", "♯");
  const map: Record<string, number> = {
    "C": 0, "C♯": 1, "D♭": 1, "D": 2, "D♯": 3, "E♭": 3, "E": 4,
    "F": 5, "F♯": 6, "G♭": 6, "G": 7, "G♯": 8, "A♭": 8, "A": 9, "A♯": 10, "B♭": 10, "B": 11,
  };
  return map[norm] ?? 0;
}

function indexToPitch(i: number, useFlats = false): string {
  const idx = ((i % 12) + 12) % 12;
  return useFlats ? FLATS[idx] : SHARPS[idx];
}

export function parseChord(label: string): ParsedChord {
  if (!label || label === "N" || label === "X") {
    return { root: null, quality: "", bass: null };
  }

  // Split slash: "G/B" -> ["G", "B"]
  const [main, bass = null] = label.split("/");

  // Pull the root (A-G with optional accidental) off the front; whatever
  // follows is the quality suffix. Tolerates both shorthand ("Cmaj7", "Cm")
  // and legacy Chordino-style colon ("C:maj7", "C:min").
  const m = main.match(/^([A-G][#b]?)(.*)$/);
  if (!m) return { root: null, quality: "", bass: null };

  const rootRaw = m[1];
  const qualRaw = m[2].startsWith(":") ? m[2].slice(1) : m[2];

  let quality = "";
  switch (qualRaw) {
    case "":
    case "maj":
      quality = "";
      break;
    case "m":
    case "min":
      quality = "m";
      break;
    case "7":
      quality = "7";
      break;
    case "maj7":
      quality = "maj7";
      break;
    case "m7":
    case "min7":
      quality = "m7";
      break;
    case "dim":
      quality = "dim";
      break;
    case "aug":
      quality = "aug";
      break;
    case "sus4":
      quality = "sus4";
      break;
    case "sus2":
      quality = "sus2";
      break;
    default:
      quality = qualRaw; // pass through anything else verbatim
  }

  return { root: rootRaw, quality, bass };
}

export function formatChord(p: ParsedChord): string {
  if (p.root === null) return "—";
  return p.bass
    ? `${p.root}${p.quality}/${p.bass}`
    : `${p.root}${p.quality}`;
}

export function transposeChord(label: string, semitones: number): string {
  const p = parseChord(label);
  if (p.root === null) return label;

  // Prefer flats when transposing down, sharps when transposing up — matches
  // how most lead sheets are notated.
  const useFlats = semitones < 0;

  const newRoot = indexToPitch(pitchToIndex(p.root) + semitones, useFlats);
  const newBass = p.bass
    ? indexToPitch(pitchToIndex(p.bass) + semitones, useFlats)
    : null;

  return formatChord({ root: newRoot, quality: p.quality, bass: newBass });
}

export function transposeKey(key: string, semitones: number): string {
  // key looks like "G major" or "E minor"
  const [root, ...rest] = key.split(" ");
  const useFlats = semitones < 0;
  const newRoot = indexToPitch(pitchToIndex(root) + semitones, useFlats);
  return `${newRoot} ${rest.join(" ")}`;
}
