"""Shared Google Gen AI client + retry/backoff + failure logging.

Both the chord-ID path (gemini_chords.py) and the lyric-transcription path
(gemini_lyrics.py) use these helpers. See the module-level comment on
_GEMINI_MODELS for the up-to-date model-id notes.
"""

from __future__ import annotations

import os
import time as _time
from pathlib import Path
from typing import Optional

# Model names change frequently: 2.5-pro was deprecated 2026-10-16, 3.5-flash
# went GA 2026-05-19, and 3.1-pro-preview is the current Pro API id. Override
# via GEMINI_MODEL env var or the per-request `gemini_model` form field, and
# bump _GEMINI_MODELS when Google releases new ones. Verify current ids via
# https://generativelanguage.googleapis.com/v1beta/models before shipping.
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
_GEMINI_MODELS = {"gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-pro-preview"}

_genai_client = None

_GEMINI_FAILURE_LOG = Path(__file__).parent / "logs" / "gemini-failures.log"


def _get_genai_client():
    """Lazy-init the Google Gen AI client. Raises if GEMINI_API_KEY is missing."""
    global _genai_client
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is not set")
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _gemini_status_code(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status code extraction from a google-genai exception."""
    for attr in ("code", "status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


def _gemini_generate_with_retry(
    client,
    *,
    model: str,
    contents,
    config,
    max_retries: int = 3,
    initial_delay: float = 1.0,
):
    """generate_content with exponential backoff on transient failures.

    Retries on 429 (rate limit) and 5xx; fails fast on 4xx client errors so a
    bad API key or malformed request doesn't eat three attempts.
    """
    delay = initial_delay
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:  # genai raises its own exception hierarchy
            last_exc = e
            status = _gemini_status_code(e)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if attempt == max_retries - 1:
                raise
            print(
                f"[gemini] attempt {attempt + 1}/{max_retries} failed "
                f"({type(e).__name__}: {e}); retrying in {delay:.1f}s"
            )
            _time.sleep(delay)
            delay *= 2
    # Unreachable — we either returned or re-raised above.
    assert last_exc is not None
    raise last_exc


def _log_gemini_failure(kind: str, exc: BaseException) -> None:
    """Append a Gemini failure with full traceback to backend/logs/gemini-failures.log.

    The terminal one-liner at the failure site is useful in the moment;
    this file is what we read later to figure out *why* an old run fell
    back. Best-effort — if the write itself fails we swallow it, so a
    logging glitch never masks the original error.
    """
    import datetime
    import traceback
    try:
        _GEMINI_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _GEMINI_FAILURE_LOG.open("a", encoding="utf-8") as f:
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            f.write(
                f"\n=== {ts} {kind} {type(exc).__name__}: {exc} ===\n"
            )
            f.write(traceback.format_exc())
    except Exception as log_exc:
        print(f"[gemini-failure-log] could not write log: {log_exc}")


def _resolve_model(override: Optional[str]) -> str:
    """Pick the model to use for a request: override if valid, else env default."""
    return override if override in _GEMINI_MODELS else _GEMINI_MODEL
