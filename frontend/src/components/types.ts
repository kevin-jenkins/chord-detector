export interface ChordSegment {
  start: number;
  end: number;
  label: string;
}

export interface LyricWord {
  word: string;
  start: number;
  end: number;
}

export interface Section {
  start: number;
  end: number;
  label: string;
}

export interface AnalysisResult {
  duration: number;
  bpm: number;
  key: string;
  beats: number[];
  chords: ChordSegment[];
  lyrics?: LyricWord[] | null;
  sections?: Section[] | null;
  // Renamed from `provider_used` now that chord detection has its own provider too.
  lyric_provider_used?: string | null;   // "gemini" | "whisper" | "whisper-fallback"
  chord_provider_used?: string | null;   // "hybrid" | "local" | "local-fallback"
  gemini_chord_set?: string[] | null;
  gemini_key?: string | null;
}
