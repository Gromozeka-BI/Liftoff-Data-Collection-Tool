"""Pick the best reference lap by leave-one-out NN-greedy error.

For every candidate lap we build a Reference and evaluate the mean NN-greedy
position error on every other lap. The candidate with the lowest mean error
is the most representative one.
"""
from __future__ import annotations

import logging

import numpy as np

from dct.localization.lap_loader import Lap

_log = logging.getLogger(__name__)


def _build_reference(lap: Lap, *, smooth_w: int = 5):
    from dct.localization import reference_builder as refbuild  # noqa: WPS433 — avoid import cycle

    return refbuild.build(lap, smooth_w=smooth_w)


def evaluate_reference_quality(
    ref_lap: Lap,
    other_laps: list[Lap],
    *,
    smooth_w: int = 5,
    chunk: int = 200,
) -> float:
    """Return the mean NN-greedy position error of ``other_laps`` against ``ref_lap``."""
    ref = _build_reference(ref_lap, smooth_w=smooth_w)
    total = 0.0
    n_total = 0
    for lap in other_laps:
        sticks_norm = ref.normalize_sticks(lap.sticks)
        for a in range(0, len(sticks_norm), chunk):
            b = min(a + chunk, len(sticks_norm))
            diff = sticks_norm[a:b, None, :] - ref.sticks_norm[None, :, :]
            d = np.sqrt(np.sum(diff * diff, axis=2))
            j_min = np.argmin(d, axis=1)
            pred_xyz = ref.pos[j_min]
            err = np.linalg.norm(pred_xyz - lap.pos[a:b], axis=1)
            total += float(np.sum(err))
            n_total += int(len(err))
    return total / max(1, n_total)


def select_best_reference(
    laps: list[Lap],
    *,
    smooth_w: int = 5,
    progress_cb=None,
) -> tuple[int, list[float]]:
    """Return ``(best_index_0based, scores_per_lap)``.

    ``progress_cb(done, total)`` is called after each lap if provided.
    """
    scores: list[float] = []
    n = len(laps)
    if n == 0:
        return -1, scores
    if n == 1:
        return 0, [0.0]

    for i, candidate in enumerate(laps):
        others = [lap for j, lap in enumerate(laps) if j != i]
        try:
            score = evaluate_reference_quality(
                candidate, others, smooth_w=smooth_w,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("Lap %d: failed to evaluate (%s)", candidate.index, exc)
            score = float("inf")
        scores.append(score)
        if progress_cb is not None:
            try:
                progress_cb(i + 1, n)
            except Exception:
                pass

    best = int(np.argmin(scores))
    return best, scores
