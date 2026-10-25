"""Per-image predictions and summary metrics from class scores.

    predictions = build_predictions(scores, targets, meta)   # (N, C) scores in the dataset's label space
    summary = summarize(predictions)
    results = ClassificationResults(summary, predictions)

decision_margin = (S_target - max_{j != target} S_j) / sqrt(2)   (Alvarez & Konkle 2024, UniReps eq. 4)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TOPK = (1, 5)
METRIC_KEYS = {"top1", "top5", "loss_mean", "dm_mean", "dm_median", "dm_min", "dm_max", "dm_kurtosis", "n_images"}


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def build_predictions(
    scores: np.ndarray, targets: np.ndarray, meta: pd.DataFrame | None = None, topk: tuple[int, ...] = TOPK
) -> pd.DataFrame:
    """One row per image: target, pred, correct@k, top-k labels/scores, target / max non-target score, decision
    margin, cross-entropy loss. `meta` (same row order) is prepended (index, path, ...)."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n, c = scores.shape
    if targets.shape != (n,) or targets.min() < 0 or targets.max() >= c:
        raise ValueError(f"targets must be (N,) ints in [0, {c}), got shape {targets.shape}")
    k_max = min(max(topk), c)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k_max]
    rows = np.arange(n)
    target_score = scores[rows, targets]
    others = scores.copy()
    others[rows, targets] = -np.inf
    max_nontarget = others.max(axis=1)
    df = pd.DataFrame({"target": targets, "pred": order[:, 0]})
    for k in topk:
        df[f"correct{k}"] = (order[:, : min(k, c)] == targets[:, None]).any(axis=1)
    df["topk_labels"] = list(order)
    df["topk_scores"] = list(np.take_along_axis(scores, order, axis=1))
    df["target_score"] = target_score
    df["max_nontarget_score"] = max_nontarget
    df["decision_margin"] = (target_score - max_nontarget) / math.sqrt(2)
    df["loss"] = -_log_softmax(scores)[rows, targets]
    if meta is not None:
        df = pd.concat([meta.reset_index(drop=True), df], axis=1)
    return df


def summarize(predictions: pd.DataFrame) -> dict:
    dm = predictions["decision_margin"]
    out = {f"top{k}": float(predictions[f"correct{k}"].mean()) for k in TOPK if f"correct{k}" in predictions}
    out.update(
        loss_mean=float(predictions["loss"].mean()),
        dm_mean=float(dm.mean()),
        dm_median=float(dm.median()),
        dm_min=float(dm.min()),
        dm_max=float(dm.max()),
        dm_kurtosis=float(dm.kurt()),  # pandas: Fisher, bias-corrected (as eval-anagrams)
        n_images=len(predictions),
    )
    return out


@dataclass
class ClassificationResults:
    """`predictions` is the source of truth; `summary` = metrics + bookkeeping (dataset, label map, ...)."""

    summary: dict
    predictions: pd.DataFrame

    def save(self, directory: str | Path) -> Path:
        """summary.json + results.parquet: also the on-disk unit of the results store."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(json.dumps(self.summary, indent=2, default=str) + "\n")
        self.predictions.to_parquet(directory / "results.parquet", index=False)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ClassificationResults:
        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text())
        return cls(summary, pd.read_parquet(directory / "results.parquet"))

    def __repr__(self) -> str:
        s = self.summary
        return (
            f"ClassificationResults(top1={s['top1']:.4f}, top5={s.get('top5', float('nan')):.4f}, "
            f"dm_mean={s['dm_mean']:.3f}, n={s['n_images']}, dataset={s.get('dataset')})"
        )
