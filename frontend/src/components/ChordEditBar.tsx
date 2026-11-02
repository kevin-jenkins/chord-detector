"use client";

import { useEffect, useRef, useState } from "react";
import { ChordSegment } from "./types";
import { formatChord, parseChord, transposeChord } from "./chord";
import { normalizeTypedLabel } from "./chordEdits";

interface Props {
  /** In insert mode, the span the new chord would take over (label = the chord it splits). */
  chord: ChordSegment;
  mode?: "edit" | "insert";
  semitones: number;
  /** Stored (untransposed) labels offered as one-click replacements. */
  quickPicks: string[];
  canDelete: boolean;
  /** Lets the page focus the field when the user starts typing a chord name. */
  inputRef?: React.RefObject<HTMLInputElement>;
  onApply: (storedLabel: string) => void;
  onDelete: () => void;
  onClose: () => void;
}

const fmtTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

/**
 * Inline editor for one selected chord: retype it, pick a replacement, or
 * delete it. In insert mode it names the chord to insert instead.
 */
export default function ChordEditBar({
  chord, mode = "edit", semitones, quickPicks, canDelete, inputRef, onApply, onDelete, onClose,
}: Props) {
  const inserting = mode === "insert";
  const shown = (label: string) =>
    label === "N" ? "N" : formatChord(parseChord(transposeChord(label, semitones)));
  const initialText = () => (inserting ? "" : shown(chord.label));
  const [text, setText] = useState(initialText);
  const [invalid, setInvalid] = useState(false);
  const ownInputRef = useRef<HTMLInputElement>(null);
  const fieldRef = inputRef ?? ownInputRef;

  // No auto-focus on selection: the page's Delete-key shortcut only works
  // while focus is outside this text box. Typing a letter focuses it.
  // Inserting has no Delete, so it goes straight to typing.
  useEffect(() => {
    setText(initialText());
    setInvalid(false);
    if (inserting) fieldRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chord.start, chord.label, semitones, inserting]);

  const apply = (typed: string) => {
    const label = normalizeTypedLabel(typed, semitones);
    if (label === null) {
      setInvalid(true);
      return;
    }
    onApply(label);
  };

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: 8,
        marginTop: 8,
        padding: "10px 12px",
        background: "var(--panel)",
        border: "1px solid var(--accent)",
        borderRadius: 8,
      }}
    >
      <span style={{ color: "var(--muted)", fontSize: 12 }}>
        {inserting
          ? `Insert at ${fmtTime(chord.start)}, replacing ${shown(chord.label)} until ${fmtTime(chord.end)}`
          : `${fmtTime(chord.start)} · ${(chord.end - chord.start).toFixed(1)}s`}
      </span>
      <input
        ref={fieldRef}
        placeholder={inserting ? "Chord" : undefined}
        value={text}
        aria-label="Chord"
        onFocus={(e) => e.target.select()}
        onChange={(e) => { setText(e.target.value); setInvalid(false); }}
        onKeyDown={(e) => {
          if (e.key === "Enter") apply(text);
          if (e.key === "Escape") onClose();
        }}
        style={{
          width: 90,
          fontFamily: "ui-monospace, monospace",
          fontSize: 16,
          padding: "3px 6px",
          background: "var(--chord-bg)",
          color: "var(--text)",
          border: `1px solid ${invalid ? "#e06c75" : "var(--border)"}`,
          borderRadius: 4,
        }}
      />
      <button onClick={() => apply(text)}>{inserting ? "Insert" : "Change"}</button>
      {quickPicks.filter((l) => l !== chord.label).map((l) => (
        <button
          key={l}
          onClick={() => onApply(l)}
          style={{ fontFamily: "ui-monospace, monospace" }}
          title={`${inserting ? "Insert" : "Change to"} ${shown(l)}`}
        >
          {shown(l)}
        </button>
      ))}
      {chord.label !== "N" && (
        <button onClick={() => onApply("N")} title="Mark as no chord">No chord</button>
      )}
      {!inserting && (
        <button
          onClick={onDelete}
          disabled={!canDelete}
          title="Remove this chord (Delete); the chord before it takes over its time"
          style={{ color: "#e06c75" }}
        >
          Delete
        </button>
      )}
      <button onClick={onClose} title="Close (Esc)" style={{ marginLeft: "auto" }}>✕</button>
      {invalid && (
        <span style={{ width: "100%", color: "#e06c75", fontSize: 12 }}>
          Not a chord. Try something like Am, F#m7, D/F#, or N for no chord.
        </span>
      )}
    </div>
  );
}
