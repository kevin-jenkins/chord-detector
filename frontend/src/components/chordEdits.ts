/**
 * Hand corrections to a detected chord timeline. Every function returns a new
 * list; adjacent segments that end up with the same label are merged.
 */

import { ChordSegment } from "./types";
import { parseChord, transposeChord } from "./chord";

export function mergeAdjacent(chords: ChordSegment[]): ChordSegment[] {
  const out: ChordSegment[] = [];
  for (const c of chords) {
    const prev = out[out.length - 1];
    if (prev && prev.label === c.label) {
      out[out.length - 1] = { ...prev, end: c.end };
    } else {
      out.push({ ...c });
    }
  }
  return out;
}

export function relabelChord(chords: ChordSegment[], index: number, label: string): ChordSegment[] {
  if (index < 0 || index >= chords.length) return chords;
  const next = chords.map((c, i) => (i === index ? { ...c, label } : c));
  return mergeAdjacent(next);
}

/**
 * Remove a chord by folding its time into the chord before it (the first
 * chord folds into the one after). That's how a misdetected passing tone
 * should disappear: the surrounding chord was really sounding the whole time.
 */
export function deleteChord(chords: ChordSegment[], index: number): ChordSegment[] {
  if (chords.length <= 1 || index < 0 || index >= chords.length) return chords;
  const next = chords.map((c) => ({ ...c }));
  const [removed] = next.splice(index, 1);
  if (index > 0) {
    next[index - 1].end = removed.end;
  } else {
    next[0].start = removed.start;
  }
  return mergeAdjacent(next);
}

/**
 * Turn what the user typed (in the key currently shown, i.e. after
 * `semitones` of transposition) into a stored label in the original key.
 * Returns null when it isn't a chord. "N", "-", "—" or empty mean no chord.
 */
export function normalizeTypedLabel(input: string, semitones: number): string | null {
  const s = input.trim();
  if (s === "" || s === "N" || s === "n" || s === "-" || s === "—") return "N";
  if (!/^[A-G][#b]?[A-Za-z0-9#b+()]*(\/[A-G][#b]?)?$/.test(s)) return null;
  const parsed = parseChord(s);
  if (parsed.root === null) return null;
  // Transpose back to the original key, then through 0 to normalize to sharps
  // like the backend's labels.
  return transposeChord(transposeChord(s, -semitones), 0);
}

/** localStorage key for edits to one specific detected result. */
export async function editsStorageKey(duration: number, detected: ChordSegment[]): Promise<string> {
  const json = JSON.stringify({ d: duration, c: detected });
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(json));
  const hex = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  return `chord-edits:v1:${hex}`;
}

export function loadEdits(key: string): ChordSegment[] | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : null;
  } catch {
    return null;
  }
}

export function saveEdits(key: string, chords: ChordSegment[] | null): void {
  try {
    if (chords) localStorage.setItem(key, JSON.stringify(chords));
    else localStorage.removeItem(key);
  } catch {
    // Storage blocked or full: edits still work for this session.
  }
}
