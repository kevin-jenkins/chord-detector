// Run with `npm test`. Covers the pure edit logic; the page wiring is checked by hand.
import { test } from "node:test";
import { deepStrictEqual as eq, strictEqual as is, notStrictEqual as isnt } from "node:assert";
import {
  deleteChord, editsStorageKey, insertChord, mergeAdjacent, normalizeTypedLabel, relabelChord,
  snapToBeat,
} from "./chordEdits";

const seg = (label: string, start: number, end: number) => ({ label, start, end });
const song = [seg("C#m", 0, 2), seg("G#m", 2, 3), seg("C#m", 3, 5), seg("E", 5, 6), seg("A", 6, 7), seg("B", 7, 8)];

test("delete passing tone folds into previous chord and merges", () => {
  eq(deleteChord(song, 1).slice(0, 2), [seg("C#m", 0, 5), seg("E", 5, 6)]);
});

test("delete first chord extends the next back to the start", () => {
  eq(deleteChord(song, 0)[0], seg("G#m", 0, 3));
});

test("delete last chord extends the previous to the end", () => {
  const out = deleteChord(song, 5);
  eq(out[out.length - 1], seg("A", 6, 8));
});

test("delete after a no-chord gap folds into the next chord, not the silence", () => {
  const gap = [seg("N", 0, 2), seg("G#m", 2, 3), seg("E", 3, 5)];
  eq(deleteChord(gap, 1), [seg("N", 0, 2), seg("E", 2, 5)]);
  // ...unless there's nothing real after it either.
  eq(deleteChord([seg("N", 0, 2), seg("G#m", 2, 3)], 1), [seg("N", 0, 3)]);
  eq(deleteChord([seg("N", 0, 2), seg("G#m", 2, 3), seg("N", 3, 4)], 1), [seg("N", 0, 4)]);
});

test("delete keeps total coverage", () => {
  for (let i = 0; i < song.length; i++) {
    const out = deleteChord(song, i);
    is(out[0].start, 0);
    is(out[out.length - 1].end, 8);
    for (let k = 1; k < out.length; k++) is(out[k].start, out[k - 1].end);
  }
});

test("delete refuses to remove the only chord", () => {
  const one = [seg("A", 0, 4)];
  is(deleteChord(one, 0), one);
});

test("relabel changes and merges with an identical neighbor", () => {
  eq(relabelChord(song, 1, "C#m").slice(0, 2), [seg("C#m", 0, 5), seg("E", 5, 6)]);
  is(relabelChord(song, 4, "Am")[4].label, "Am");
});

test("insert splits the chord at the given time", () => {
  eq(insertChord(song, 4, "B").slice(2, 5), [seg("C#m", 3, 4), seg("B", 4, 5), seg("E", 5, 6)]);
});

test("insert twice puts a short missed chord in the middle of a long one", () => {
  const long = [seg("E", 0, 8)];
  eq(insertChord(insertChord(long, 4, "A"), 6, "E"), [seg("E", 0, 4), seg("A", 4, 6), seg("E", 6, 8)]);
});

test("insert at a chord's start relabels it instead of leaving a sliver", () => {
  eq(insertChord(song, 5.01, "F#m")[3], seg("F#m", 5, 6));
});

test("insert merges with a matching neighbor and is a no-op for the same chord", () => {
  eq(insertChord(song, 2.5, "C#m").slice(0, 1), [seg("C#m", 0, 2)]);
  eq(insertChord(song, 2.5, "C#m")[1], seg("G#m", 2, 2.5));
  eq(insertChord(song, 4, "C#m"), song);
});

test("insert outside the song changes nothing", () => {
  is(insertChord(song, 9, "D"), song);
  is(insertChord(song, -1, "D"), song);
});

test("snap moves to the nearest beat within half a second", () => {
  const beats = [0.5, 1.0, 1.5, 2.0];
  is(snapToBeat(1.2, beats), 1.0);
  is(snapToBeat(1.3, beats), 1.5);
  is(snapToBeat(3.0, beats), 3.0);
  is(snapToBeat(1.2, []), 1.2);
});

test("inputs are not mutated", () => {
  const copy = JSON.parse(JSON.stringify(song));
  deleteChord(song, 2);
  relabelChord(song, 2, "D");
  insertChord(song, 4, "B");
  mergeAdjacent(song);
  eq(song, copy);
});

test("typed labels: plain, no-chord, invalid", () => {
  is(normalizeTypedLabel("Am", 0), "Am");
  is(normalizeTypedLabel(" F#m7 ", 0), "F#m7");
  is(normalizeTypedLabel("Bb", 0), "A#"); // sharps, like the backend
  is(normalizeTypedLabel("D/F#", 0), "D/F#");
  is(normalizeTypedLabel("Cadd9", 0), "Cadd9");
  is(normalizeTypedLabel("N", 0), "N");
  is(normalizeTypedLabel("", 0), "N");
  is(normalizeTypedLabel("H", 0), null);
  is(normalizeTypedLabel("hello", 0), null);
  is(normalizeTypedLabel("A/Q", 0), null);
});

test("typed labels: lowercase root and NC spellings", () => {
  is(normalizeTypedLabel("am", 0), "Am");
  is(normalizeTypedLabel("bb", 0), "A#");
  is(normalizeTypedLabel("f#m7", 0), "F#m7");
  for (const nc of ["NC", "nc", "N.C.", "n.c.", "N/C"]) is(normalizeTypedLabel(nc, 0), "N", nc);
});

test("typed labels: roots the transposer can't represent are rejected, not turned into C", () => {
  for (const bad of ["Cb", "Fb", "E#", "B#", "Cbm", "E#m7", "A/Cb", "D/E#"]) {
    is(normalizeTypedLabel(bad, 0), null, bad);
  }
});

test("typed labels are stored in the original key when transposed", () => {
  is(normalizeTypedLabel("Am", 2), "Gm"); // shown +2: typed Am was Gm
  is(normalizeTypedLabel("Cm", -1), "C#m"); // shown -1: typed Cm was C#m
  is(normalizeTypedLabel("E/G#", 3), "C#/F");
});

test("storage key depends only on the audio bytes", async () => {
  const a = await editsStorageKey(new Blob([new Uint8Array([1, 2, 3])]));
  is(a, await editsStorageKey(new Blob([new Uint8Array([1, 2, 3])])));
  isnt(a, await editsStorageKey(new Blob([new Uint8Array([1, 2, 4])])));
  is(a.startsWith("chord-edits:v2:"), true);
});
