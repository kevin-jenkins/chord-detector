"use client";

import { useEffect, useRef } from "react";
import { ChordSegment } from "./types";
import { parseChord, formatChord, transposeChord } from "./chord";

interface Props {
  chords: ChordSegment[];
  duration: number;
  currentTime: number;
  semitones: number;
  onSeek: (time: number) => void;
}

/**
 * Two-row chord display:
 *   Top row: large "current chord" + "next chord" preview
 *   Bottom row: scrollable strip of all chord blocks, sized by duration,
 *               with the active chord highlighted and auto-scrolled into view.
 */
export default function ChordTimeline({
  chords,
  duration,
  currentTime,
  semitones,
  onSeek,
}: Props) {
  const stripRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLDivElement>(null);

  const activeIdx = chords.findIndex(
    (c) => currentTime >= c.start && currentTime < c.end
  );
  const active = activeIdx >= 0 ? chords[activeIdx] : null;
  const next = activeIdx >= 0 && activeIdx + 1 < chords.length
    ? chords[activeIdx + 1]
    : null;

  // Auto-scroll the active chord into view in the strip
  useEffect(() => {
    if (activeRef.current && stripRef.current) {
      const el = activeRef.current;
      const container = stripRef.current;
      const elLeft = el.offsetLeft;
      const elRight = elLeft + el.offsetWidth;
      const viewLeft = container.scrollLeft;
      const viewRight = viewLeft + container.clientWidth;
      if (elLeft < viewLeft || elRight > viewRight) {
        container.scrollTo({
          left: elLeft - container.clientWidth / 2,
          behavior: "smooth",
        });
      }
    }
  }, [activeIdx]);

  const displayLabel = (label: string) =>
    formatChord(parseChord(transposeChord(label, semitones)));

  // Skip "N" (no-chord) for the big display
  const bigLabel = active && active.label !== "N" ? displayLabel(active.label) : "—";
  const nextLabel = next && next.label !== "N" ? displayLabel(next.label) : "—";

  return (
    <div style={{ width: "100%" }}>
      {/* Current / next preview */}
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 32,
          padding: "24px 16px",
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          marginBottom: 16,
        }}
      >
        <div>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>NOW</div>
          <div
            style={{
              fontSize: 64,
              fontWeight: 700,
              fontFamily: "ui-monospace, SFMono-Regular, monospace",
              color: "var(--accent-bright)",
              lineHeight: 1,
            }}
          >
            {bigLabel}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>NEXT</div>
          <div
            style={{
              fontSize: 32,
              fontWeight: 500,
              fontFamily: "ui-monospace, SFMono-Regular, monospace",
              color: "var(--muted)",
              lineHeight: 1,
            }}
          >
            {nextLabel}
          </div>
        </div>
      </div>

      {/* Scrollable chord strip */}
      <div
        ref={stripRef}
        style={{
          display: "flex",
          overflowX: "auto",
          gap: 4,
          padding: 8,
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          minHeight: 90,
        }}
      >
        {chords.map((c, i) => {
          const dur = c.end - c.start;
          // 60px per second feels right at a typical viewport width; clamp so
          // very long sustained chords don't become unusably wide.
          const width = Math.max(48, Math.min(240, dur * 60));
          const isActive = i === activeIdx;
          const isSilent = c.label === "N";

          return (
            <div
              key={i}
              ref={isActive ? activeRef : null}
              onClick={() => onSeek(c.start)}
              style={{
                flex: `0 0 ${width}px`,
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
                background: isActive ? "var(--chord-active)" : "var(--chord-bg)",
                border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
                borderRadius: 6,
                padding: "10px 8px",
                cursor: "pointer",
                opacity: isSilent ? 0.4 : 1,
                transition: "background 0.15s, border-color 0.15s",
              }}
            >
              <div
                style={{
                  fontSize: 20,
                  fontWeight: 600,
                  fontFamily: "ui-monospace, SFMono-Regular, monospace",
                  color: isActive ? "var(--accent-bright)" : "var(--text)",
                }}
              >
                {isSilent ? "—" : displayLabel(c.label)}
              </div>
              <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
                {dur.toFixed(1)}s
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
