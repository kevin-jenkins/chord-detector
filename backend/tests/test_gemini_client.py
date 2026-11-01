"""Tests for the gemini_client retry / resolve / status-code logic.

No network — exceptions are faked and the generate_content call site is
mocked out so we're exercising only the pure branching logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gemini_client import (
    _GEMINI_MODEL,
    _GEMINI_MODELS,
    _gemini_generate_with_retry,
    _gemini_status_code,
    _resolve_model,
)


# ---- _resolve_model -------------------------------------------------------

class TestResolveModel:
    def test_valid_override_wins(self):
        # Pick any valid model that isn't the default.
        other = next(m for m in _GEMINI_MODELS if m != _GEMINI_MODEL)
        assert _resolve_model(other) == other

    def test_invalid_override_falls_back_to_default(self):
        assert _resolve_model("gemini-nope-9000") == _GEMINI_MODEL

    def test_none_falls_back_to_default(self):
        assert _resolve_model(None) == _GEMINI_MODEL


# ---- _gemini_status_code --------------------------------------------------

class TestGeminiStatusCode:
    def test_code_attr(self):
        exc = Exception("x")
        exc.code = 429  # type: ignore[attr-defined]
        assert _gemini_status_code(exc) == 429

    def test_status_code_attr(self):
        exc = Exception("x")
        exc.status_code = 500  # type: ignore[attr-defined]
        assert _gemini_status_code(exc) == 500

    def test_no_code_returns_none(self):
        assert _gemini_status_code(Exception("x")) is None

    def test_non_int_ignored(self):
        exc = Exception("x")
        exc.code = "429"  # type: ignore[attr-defined]
        assert _gemini_status_code(exc) is None


# ---- _gemini_generate_with_retry ------------------------------------------

def _mock_client(side_effects):
    """Build a client whose generate_content yields these side effects in order."""
    client = MagicMock()
    client.models.generate_content = MagicMock(side_effect=side_effects)
    return client


def _status_err(code: int) -> Exception:
    e = Exception(f"HTTP {code}")
    e.code = code  # type: ignore[attr-defined]
    return e


class TestGeminiRetry:
    def test_success_first_try(self):
        client = _mock_client(["ok"])
        got = _gemini_generate_with_retry(
            client, model="m", contents=[], config=None, timeout_sec=5,
        )
        assert got == "ok"
        assert client.models.generate_content.call_count == 1

    def test_429_retries_then_succeeds(self):
        client = _mock_client([_status_err(429), "ok"])
        got = _gemini_generate_with_retry(
            client, model="m", contents=[], config=None,
            max_retries=3, initial_delay=0.0, timeout_sec=5,
        )
        assert got == "ok"
        assert client.models.generate_content.call_count == 2

    def test_500_retries_then_succeeds(self):
        client = _mock_client([_status_err(503), "ok"])
        got = _gemini_generate_with_retry(
            client, model="m", contents=[], config=None,
            max_retries=3, initial_delay=0.0, timeout_sec=5,
        )
        assert got == "ok"

    def test_400_fails_fast(self):
        client = _mock_client([_status_err(400)])
        with pytest.raises(Exception) as exc_info:
            _gemini_generate_with_retry(
                client, model="m", contents=[], config=None,
                max_retries=3, initial_delay=0.0, timeout_sec=5,
            )
        assert exc_info.value.code == 400  # type: ignore[attr-defined]
        assert client.models.generate_content.call_count == 1

    def test_401_fails_fast(self):
        """Bad API key must not eat retries."""
        client = _mock_client([_status_err(401)])
        with pytest.raises(Exception):
            _gemini_generate_with_retry(
                client, model="m", contents=[], config=None,
                max_retries=3, initial_delay=0.0, timeout_sec=5,
            )
        assert client.models.generate_content.call_count == 1

    def test_404_fails_fast(self):
        """Wrong model name must not eat retries."""
        client = _mock_client([_status_err(404)])
        with pytest.raises(Exception):
            _gemini_generate_with_retry(
                client, model="m", contents=[], config=None,
                max_retries=3, initial_delay=0.0, timeout_sec=5,
            )
        assert client.models.generate_content.call_count == 1

    def test_exhausts_retries(self):
        client = _mock_client([_status_err(500)] * 3)
        with pytest.raises(Exception):
            _gemini_generate_with_retry(
                client, model="m", contents=[], config=None,
                max_retries=3, initial_delay=0.0, timeout_sec=5,
            )
        assert client.models.generate_content.call_count == 3

    def test_unknown_exception_retries(self):
        """Exceptions without a status code (e.g., network errors) retry."""
        client = _mock_client([ConnectionError("boom"), "ok"])
        got = _gemini_generate_with_retry(
            client, model="m", contents=[], config=None,
            max_retries=3, initial_delay=0.0, timeout_sec=5,
        )
        assert got == "ok"
        assert client.models.generate_content.call_count == 2

    def test_timeout_raises_and_counts_as_retry(self):
        """A hung call hits the deadline, gets wrapped in TimeoutError, retries."""
        import time

        call_count = {"n": 0}

        def slow_then_fast(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                time.sleep(1.0)  # exceeds the 0.1s timeout below
                return "late"
            return "ok"

        client = MagicMock()
        client.models.generate_content = slow_then_fast
        got = _gemini_generate_with_retry(
            client, model="m", contents=[], config=None,
            max_retries=3, initial_delay=0.0, timeout_sec=0.1,
        )
        assert got == "ok"
