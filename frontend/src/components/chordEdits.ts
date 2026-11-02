/**
 * Hand corrections to a detected chord timeline. Every function returns a new
 * list; adjacent segments that end up with the same label are merged.
 */

import { ChordSegment } from "./types";
import { transposeChord } from "./chord";

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
 * Remove a chord by folding its time into the chord before it. That's how a
 * misdetected passing tone should disappear: the surrounding chord was really
 * sounding the whole time. The first chord, or one that follows a no-chord
 * gap, folds into the chord after it instead.
 */
export function deleteChord(chords: ChordSegment[], index: number): ChordSegment[] {
  if (chords.length <= 1 || index < 0 || index >= chords.length) return chords;
  const next = chords.map((c) => ({ ...c }));
  const [removed] = next.splice(index, 1);
  const prev = index > 0 ? next[index - 1] : null;
  const after = index < next.length ? next[index] : null;
  if (prev && (prev.label !== "N" || !after || after.label === "N")) {
    prev.end = removed.end;
  } else if (after) {
    after.start = removed.start;
  }
  return mergeAdjacent(next);
}

// A split closer than this to a chord's start just relabels that chord rather
// than leaving a sliver of the old one.
const MIN_SPLIT = 0.05;

/**
 * Insert a chord change at `time`: the chord sounding there is split, and
 * `label` runs from `time` to where that chord ended. A missed short chord is
 * two inserts, one at its start and one back to the old chord at its end.
 */
export function insertChord(chords: ChordSegment[], time: number, label: string): ChordSegment[] {
  const index = chords.findIndex((c) => time >= c.start && time < c.end);
  if (index < 0) return chords;
  const c = chords[index];
  if (time - c.start < MIN_SPLIT) return relabelChord(chords, index, label);
  const next = [
    ...chords.slice(0, index),
    { ...c, end: time },
    { start: time, end: c.end, label },
    ...chords.slice(index + 1),
  ];
  return mergeAdjacent(next);
}

/**
 * Move a time picked by ear (the playhead) onto the nearest detected beat, as
 * the detector itself does, unless no beat is within half a second.
 */
export function snapToBeat(time: number, beats: number[]): number {
  let best = time;
  let bestDist = 0.5;
  for (const b of beats) {
    const d = Math.abs(b - time);
    if (d <= bestDist) {
      best = b;
      bestDist = d;
    }
  }
  return best;
}

/**
 * The beat before (direction -1) or after (+1) `time`, for nudging an insert
 * point. Steps a quarter second when there are no beats; null at either end.
 */
export function nudgeToBeat(time: number, beats: number[], direction: -1 | 1, duration: number): number | null {
  const EPS = 0.01;
  let t: number | undefined;
  if (beats.length === 0) {
    t = time + direction * 0.25;
  } else if (direction < 0) {
    t = [...beats].reverse().find((b) => b < time - EPS);
  } else {
    t = beats.find((b) => b > time + EPS);
  }
  return t !== undefined && t > 0 && t < duration ? t : null;
}

// Label shown on the strip for an insert whose chord hasn't been picked yet.
export const PENDING_LABEL = "?";

// Every root and bass note chord.ts can transpose. Anything else (Cb, E#, …)
// would silently come out as C, so the quality can't start with # or b.
const NOTE = "(?:[ACDFG]#?|[BDEGA]b|[BE])";
const TYPED_CHORD = new RegExp(`^${NOTE}(?:[ac-zA-Z0-9+()][A-Za-z0-9#b+()]*)?(?:/${NOTE})?$`);
const NO_CHORD = new Set(["", "N", "NC", "N.C.", "N/C", "-", "—"]);

/**
 * Turn what the user typed (in the key currently shown, i.e. after
 * `semitones` of transposition) into a stored label in the original key.
 * Returns null when it isn't a chord. "N", "NC", "-" or empty mean no chord.
 */
export function normalizeTypedLabel(input: string, semitones: number): string | null {
  let s = input.trim();
  if (NO_CHORD.has(s.toUpperCase())) return "N";
  s = s.charAt(0).toUpperCase() + s.slice(1);
  if (!TYPED_CHORD.test(s)) return null;
  // Transpose back to the original key, then through 0 to normalize to sharps
  // like the backend's labels.
  return transposeChord(transposeChord(s, -semitones), 0);
}

/**
 * localStorage key for edits to one audio file. Keyed by the file's bytes, not
 * the detected chords, so corrections survive re-analysis with other settings
 * or a newer matcher.
 */
export async function editsStorageKey(audio: Blob): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", await audio.arrayBuffer());
  const hex = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  return `chord-edits:v2:${hex}`;
}

interface SavedEdits {
  duration: number;
  chords: ChordSegment[];
}

/**
 * Saved edits for this file, or null. Also null if the saved timeline was for
 * a different duration (a decoder change, say), since it wouldn't line up.
 */
export function loadEdits(key: string, duration: number): ChordSegment[] | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const saved: SavedEdits = JSON.parse(raw);
    if (typeof saved?.duration !== "number" || Math.abs(saved.duration - duration) > 0.5) return null;
    return Array.isArray(saved.chords) && saved.chords.length > 0 ? saved.chords : null;
  } catch {
    return null;
  }
}

/** Returns false if the browser refused to store them. */
export function saveEdits(key: string, duration: number, chords: ChordSegment[] | null): boolean {
  try {
    if (chords) localStorage.setItem(key, JSON.stringify({ duration, chords } satisfies SavedEdits));
    else localStorage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}
