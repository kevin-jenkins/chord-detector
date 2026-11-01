"""Pydantic response models shared across modules."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ChordSegment(BaseModel):
    start: float
    end: float
    label: str


class LyricWord(BaseModel):
    word: str
    start: float
    end: float


class Section(BaseModel):
    start: float
    end: float
    label: str   # e.g. "Intro", "Verse 1", "Chorus", "Bridge", "Outro"


class AnalysisResult(BaseModel):
    duration: float
    bpm: float
    key: str
    beats: List[float]
    chords: List[ChordSegment]
    lyrics: Optional[List[LyricWord]] = None
    sections: Optional[List[Section]] = None
    lyric_provider_used: Optional[str] = None   # "gemini" | "whisper" | "whisper-fallback"
    chord_provider_used: Optional[str] = None   # "hybrid" | "local" | "local-fallback"
    gemini_chord_set: Optional[List[str]] = None
    gemini_key: Optional[str] = None
