"""Stem separation (demucs).

Drops drums for chord recognition and keeps bass + other for the harmonic
mix. Vocals are extracted separately for lyric transcription. First call
downloads the model weights to ~/.cache/torch/hub.

Model is configurable via the DEMUCS_MODEL env var:
  mdx_extra_q  (default) — quantized MDX-Net, fastest on CPU, mild quality drop
  htdemucs_ft            — fine-tuned Hybrid Transformer Demucs, ~2x slower on
                           CPU with noticeably better vocal isolation. The right
                           default when you have a GPU.
Device probe is cached for the process; MPS falls back permanently to CPU
if a run fails, since repeated flapping is worse than the one-time cost.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

from cache import _stems_cache_get, _stems_cache_put

_demucs_model = None    # cached across requests
_demucs_device = None   # resolved on first stem-separation request
_DEMUCS_MODEL_NAME = os.environ.get("DEMUCS_MODEL", "mdx_extra_q")


def _get_demucs_device():
    """Pick the best inference device for demucs once per process.

    Order of preference:
      1. CUDA — NVIDIA discrete GPUs on Linux/Windows.
      2. MPS — Apple Silicon (M1/M2/M3), with CPU fallback enabled for ops
         Metal doesn't implement.
      3. CPU — universal fallback.
    """
    global _demucs_device
    if _demucs_device is not None:
        return _demucs_device
    try:
        import torch
        if torch.cuda.is_available():
            _demucs_device = "cuda"
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "unknown"
            print(f"Demucs device: CUDA ({name})")
            return _demucs_device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            _demucs_device = "mps"
            print("Demucs device: MPS (Apple Silicon, CPU fallback enabled)")
            return _demucs_device
    except Exception as e:
        print(f"GPU detection failed ({type(e).__name__}: {e}); using CPU")
    _demucs_device = "cpu"
    print("Demucs device: CPU")
    return _demucs_device


def run_stem_separation(
    audio_path: str, want_vocals: bool = False,
    audio_hash: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Run demucs and return (harm_mono, vocals_mono_or_None, sr).

    Stems sub-cache: when `audio_hash` matches a previous run on the same
    audio bytes, return the cached stems directly and skip demucs entirely.
    """
    if audio_hash is not None:
        hit = _stems_cache_get(audio_hash, need_vocals=want_vocals)
        if hit is not None:
            harm_cached, vocals_cached, sr_cached = hit
            print(f"[stems-cache] hit for {audio_hash[:12]}...")
            return (
                harm_cached,
                vocals_cached if want_vocals else None,
                sr_cached,
            )

    global _demucs_model, _demucs_device
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    device = _get_demucs_device()

    # Windows-specific: PyTorch's CPU thread auto-detection under-counts
    # E-cores on 12th-gen+ Intel hybrid CPUs. Force the full physical count.
    if sys.platform == "win32" and device == "cpu":
        try:
            cpu_threads = max(1, os.cpu_count() or 1)
            torch.set_num_threads(cpu_threads)
        except Exception:
            pass

    if _demucs_model is None:
        print(f"Loading demucs model: {_DEMUCS_MODEL_NAME}")
        m = get_model(_DEMUCS_MODEL_NAME)
        m.to(device).eval()
        _demucs_model = m

    model = _demucs_model
    sr = int(model.samplerate)

    wav = AudioFile(audio_path).read(
        streams=0, samplerate=sr, channels=model.audio_channels
    )

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    wav_norm = (wav - mean) / (std + 1e-8)

    with torch.no_grad():
        def _run(dev):
            return apply_model(
                model, wav_norm[None],
                device=dev, progress=False,
                shifts=0,
                overlap=0.1,
            )[0]

        try:
            sources = _run(device)
        except RuntimeError as e:
            if device == "cpu":
                raise
            print(
                f"{device} inference failed ({type(e).__name__}: {e}); "
                f"falling back to CPU for the rest of the process"
            )
            _demucs_device = "cpu"
            model.to("cpu")
            sources = _run("cpu")
    sources = sources * std + mean

    src_names = list(model.sources)
    bass = sources[src_names.index("bass")]
    other = sources[src_names.index("other")]
    harm_mono = (bass + other).mean(dim=0).cpu().numpy().astype(np.float32)

    # Always extract vocals when we have a cache key, so future requests on
    # the same audio that DO want lyrics hit the stems cache cleanly.
    vocals_mono: Optional[np.ndarray] = None
    if want_vocals or audio_hash is not None:
        vocals = sources[src_names.index("vocals")]
        vocals_mono = vocals.mean(dim=0).cpu().numpy().astype(np.float32)

    if audio_hash is not None:
        _stems_cache_put(audio_hash, harm_mono, vocals_mono, sr)

    return harm_mono, vocals_mono if want_vocals else None, sr
