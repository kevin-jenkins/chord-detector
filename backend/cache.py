"""Two-tier result cache + in-memory stems cache.

Result cache keyed by sha256(audio_bytes + sorted params). In-memory LRU on
top of an on-disk JSON tier that survives restart. Stems sub-cache keyed by
sha256(audio_bytes alone) — stems depend only on the audio, so flipping any
request param doesn't invalidate them. Single-user local app, so the shared
state here isn't a concurrency problem; /analyze is already serialized by
a lock in main.
"""

from __future__ import annotations

import hashlib
import json as _json
from pathlib import Path

_RESULT_CACHE: dict = {}
_RESULT_CACHE_MAX = 32
_RESULT_CACHE_DIR = Path(__file__).parent / ".cache" / "results"
_RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DISK_CACHE_MAX = 500   # prune oldest-mtime entries past this on each write

_STEMS_CACHE: dict = {}
_STEMS_CACHE_MAX = 4


def _cache_key(audio_bytes: bytes, **params) -> str:
    h = hashlib.sha256()
    h.update(audio_bytes)
    param_str = "|".join(f"{k}={params[k]}" for k in sorted(params))
    h.update(param_str.encode("utf-8"))
    return h.hexdigest()


def _audio_hash(audio_bytes: bytes) -> str:
    return hashlib.sha256(audio_bytes).hexdigest()


def _cache_get(key: str):
    # Memory first
    val = _RESULT_CACHE.pop(key, None)
    if val is not None:
        _RESULT_CACHE[key] = val  # touch LRU
        return val
    # Disk fall-through
    disk_path = _RESULT_CACHE_DIR / f"{key}.json"
    if disk_path.exists():
        try:
            with open(disk_path, "r", encoding="utf-8") as f:
                val = _json.load(f)
            _RESULT_CACHE[key] = val  # hydrate memory
            print(f"[cache] disk hit for {key[:12]}...")
            return val
        except Exception as e:
            print(f"[cache] disk read failed for {key[:12]}...: {e}")
    return None


def _cache_put(key: str, value) -> None:
    # Memory tier
    _RESULT_CACHE[key] = value
    while len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
        oldest = next(iter(_RESULT_CACHE))
        _RESULT_CACHE.pop(oldest, None)
    # Disk tier — best-effort; never let a write failure break the request
    try:
        tmp = _RESULT_CACHE_DIR / f".{key}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(value, f)
        tmp.replace(_RESULT_CACHE_DIR / f"{key}.json")
    except Exception as e:
        print(f"[cache] disk write failed for {key[:12]}...: {e}")
    _prune_disk_cache()


def _prune_disk_cache(limit: int = _DISK_CACHE_MAX) -> None:
    """Trim the on-disk result cache to `limit` newest-mtime entries."""
    try:
        entries = [(p.stat().st_mtime, p) for p in _RESULT_CACHE_DIR.glob("*.json")]
        if len(entries) <= limit:
            return
        entries.sort()  # oldest first
        for _, p in entries[: len(entries) - limit]:
            try:
                p.unlink()
            except OSError:
                pass
    except Exception as e:
        print(f"[cache] disk prune failed: {e}")


def _stems_cache_get(key: str, need_vocals: bool):
    entry = _STEMS_CACHE.get(key)
    if entry is None:
        return None
    _, vocals, _ = entry
    if need_vocals and vocals is None:
        # Half-populated entry: don't touch LRU, let something else take
        # the slot when the recompute path puts a full entry back.
        return None
    # True hit — refresh LRU position by re-inserting at the tail.
    _STEMS_CACHE.pop(key)
    _STEMS_CACHE[key] = entry
    return entry


def _stems_cache_put(key: str, harm, vocals, sr) -> None:
    _STEMS_CACHE[key] = (harm, vocals, sr)
    while len(_STEMS_CACHE) > _STEMS_CACHE_MAX:
        oldest = next(iter(_STEMS_CACHE))
        _STEMS_CACHE.pop(oldest, None)
