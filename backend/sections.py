"""Song structure: boundary detection on beat-synced chroma, then KMeans
cluster the inter-boundary segments and heuristically name clusters as
Verse / Chorus / Bridge / Intro / Outro.

Returns an empty list when there aren't enough beats to cluster meaningfully.
Adjacent segments sharing the same emitted label are merged on exit.
"""

from __future__ import annotations

from typing import List

import librosa
import numpy as np

from models import Section


def detect_sections(
    y_harm: np.ndarray, sr: int, duration: float, beat_frames: np.ndarray
) -> List[Section]:
    from sklearn.cluster import KMeans

    n_beats = len(beat_frames)
    if n_beats < 16:
        return []

    chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr)
    beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

    # Choose k roughly proportional to song length, capped for short songs.
    k_segs = max(3, min(8, n_beats // 16))
    bounds = librosa.segment.agglomerative(beat_chroma, k=k_segs)
    bounds = np.asarray(bounds, dtype=int)

    edges = np.unique(np.concatenate([[0], bounds, [n_beats]]))
    if len(edges) < 2:
        return []

    seg_feats = []
    seg_ranges = []
    for i in range(len(edges) - 1):
        s, e = int(edges[i]), int(edges[i + 1])
        if e <= s:
            continue
        feat = beat_chroma[:, s:e].mean(axis=1)
        seg_feats.append(feat)
        seg_ranges.append((s, e))

    if len(seg_feats) < 2:
        return []

    feats = np.stack(seg_feats)
    n_clusters = min(4, len(seg_feats))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    cluster_ids = km.fit_predict(feats).tolist()

    counts: dict = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1

    sorted_clusters = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    # Verse/Chorus naming only makes sense when TWO clusters recur. On a
    # chordally uniform song (one cluster owns most of the timeline), fall
    # back to neutral "Section A/B/…" labels.
    has_two_recurring = (
        sorted_clusters[0][1] >= 2
        and len(sorted_clusters) > 1
        and sorted_clusters[1][1] >= 2
    )

    if has_two_recurring:
        chorus_id = sorted_clusters[0][0]
        verse_id = sorted_clusters[1][0]
        # Pop convention: the earliest-appearing recurring cluster is usually
        # the verse; swap if the "chorus" candidate first appears before it.
        first_chorus = next(i for i, c in enumerate(cluster_ids) if c == chorus_id)
        first_verse = next(i for i, c in enumerate(cluster_ids) if c == verse_id)
        if first_chorus < first_verse:
            chorus_id, verse_id = verse_id, chorus_id

        base_names: List[str] = []
        for idx, cid in enumerate(cluster_ids):
            if cid == chorus_id:
                base_names.append("Chorus")
            elif cid == verse_id:
                base_names.append("Verse")
            elif idx == 0:
                base_names.append("Intro")
            elif idx == len(cluster_ids) - 1:
                base_names.append("Outro")
            else:
                base_names.append("Bridge")
    else:
        letter_for_cluster = {
            cid: chr(ord("A") + i) for i, (cid, _) in enumerate(sorted_clusters)
        }
        base_names = [
            f"Section {letter_for_cluster[cid]}" for cid in cluster_ids
        ]
        if counts[cluster_ids[0]] == 1:
            base_names[0] = "Intro"
        if counts[cluster_ids[-1]] == 1:
            base_names[-1] = "Outro"

    counts_by_label: dict = {}
    for n in base_names:
        counts_by_label[n] = counts_by_label.get(n, 0) + 1

    seen: dict = {}
    final_labels: List[str] = []
    for n in base_names:
        if counts_by_label[n] == 1:
            final_labels.append(n)
        else:
            seen[n] = seen.get(n, 0) + 1
            final_labels.append(f"{n} {seen[n]}")

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    raw: List[tuple] = []
    for (s_beat, e_beat), label in zip(seg_ranges, final_labels):
        start_t = float(beat_times[s_beat]) if s_beat < len(beat_times) else 0.0
        end_t = (
            float(beat_times[e_beat])
            if e_beat < len(beat_times)
            else duration
        )
        start_t = max(0.0, min(start_t, duration))
        end_t = max(start_t, min(end_t, duration))
        raw.append((start_t, end_t, label))

    merged: List[tuple] = []
    for s, e, label in raw:
        if merged and merged[-1][2] == label:
            ps, _, pl = merged[-1]
            merged[-1] = (ps, e, pl)
        else:
            merged.append((s, e, label))

    return [Section(start=s, end=e, label=l) for s, e, l in merged]
