"""classification_eval: run a model over a labelled image set and score it.

from visionlab.evals.classification import classification_eval
results = classification_eval(model, transforms.val, "in1k-val")
results.summary["top1"]; results.predictions      # per image, incl. decision margins
"""

from __future__ import annotations

import getpass
import platform
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm.auto import tqdm
from visionlab.evals.core.spec import transform_signature

from .data import DEFAULT_DATASET, EvalDataset, decode, load_eval_dataset
from .labels import label_map as make_label_map
from .scoring import TOPK, ClassificationResults, build_predictions, summarize
from .version import __version__


def _resolve_device(model, device) -> torch.device:
    if device is not None:
        return torch.device(device)
    params = list(model.parameters()) if isinstance(model, torch.nn.Module) else []
    if params:
        return params[0].device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_batches(ds: EvalDataset, batch_size: int, num_workers: int):
    """(images, labels, index, paths) batches in dataset order. Raw samples are read in this process and decoded +
    transformed by a thread pool (PIL releases the GIL), one batch ahead. Not a DataLoader: reading a slipstream
    store from forked worker processes is pathologically slow (vesper 2026-10-25: < 1 img/s per worker)."""
    transform = ds.transform

    def prepare(raw):
        image = decode(raw)
        return transform(image) if transform is not None else image

    def read(start):
        idx = list(range(start, min(start + batch_size, len(ds))))
        samples = [ds.base[i] for i in idx]
        images = list(pool.map(prepare, [s["image"] for s in samples]))
        labels = [int(s[ds.spec.label_field]) for s in samples]
        return torch.stack(images), torch.tensor(labels), np.asarray(idx), [s["path"] for s in samples]

    with ThreadPoolExecutor(max(1, num_workers)) as pool, ThreadPoolExecutor(1) as ahead:
        starts = list(range(0, len(ds), batch_size))
        future = ahead.submit(read, starts[0]) if starts else None
        for k in range(len(starts)):
            batch = future.result()
            if k + 1 < len(starts):
                future = ahead.submit(read, starts[k + 1])
            yield batch


@torch.inference_mode()
def predict_scores(
    model,
    ds: EvalDataset,
    map_scores: Callable,
    batch_size: int = 128,
    device=None,
    num_workers: int = 16,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, list]:
    """(scores (N, C) in the dataset's label space, targets (N,), paths) in dataset order. `num_workers` = decode
    threads."""
    device = _resolve_device(model, device)
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
    scores, targets, paths = None, np.full(len(ds), -1, dtype=np.int64), [None] * len(ds)
    n_batches = -(-len(ds) // batch_size)
    batches = iter_batches(ds, batch_size, num_workers)
    for images, labels, idx, path in tqdm(batches, total=n_batches, disable=not progress, desc=ds.spec.name):
        out = map_scores(model(images.to(device, non_blocking=True))).detach().float().cpu().numpy()
        if scores is None:
            scores = np.full((len(ds), out.shape[1]), np.nan, dtype=np.float32)
        scores[idx], targets[idx] = out, labels.numpy()
        for i, p in zip(idx, path):
            paths[i] = p
    assert scores is not None and not np.isnan(scores).any() and (targets >= 0).all(), "some images were never scored"
    return scores, targets, paths


def classification_eval(
    model,
    transform: Callable,
    dataset: str | EvalDataset = DEFAULT_DATASET,
    label_space: str | None = None,
    batch_size: int = 128,
    device=None,
    num_workers: int = 16,
    progress: bool = True,
    **meta,
) -> ClassificationResults:
    """Top-1/top-5 + decision margins of `model` on a visionlab.datasets eval set.

    Args:
        model: (B, 3, H, W) -> (B, N) class scores in `label_space`.
        transform: PIL -> tensor, the model's own eval preprocessing (visionlab `transforms.val` = standard ImageNet eval).
        dataset: 'in1k-val' | 'in100-val' (see data.DATASETS) or a loaded EvalDataset.
        label_space: the readout's label space ('in1k', 'in100'); default = the dataset's. An in1k readout on
            in100-val is restricted to the 100 classes' columns (labels.label_map).
        **meta: bookkeeping recorded in the summary.
    """
    ds = load_eval_dataset(dataset) if isinstance(dataset, str) else dataset
    lm = make_label_map(label_space or ds.spec.label_space, ds.spec.label_space)
    resolved = _resolve_device(model, device)
    scores, targets, paths = predict_scores(
        model, ds.with_transform(transform), lm.apply, batch_size, resolved, num_workers, progress
    )
    import pandas as pd

    predictions = build_predictions(scores, targets, pd.DataFrame({"index": range(len(ds)), "path": paths}))
    summary = {
        **summarize(predictions),
        "dataset": ds.spec.name,
        "dataset_source": f"{ds.spec.source}/{ds.spec.split}/{ds.spec.fmt}",
        "dataset_revision": ds.revision,
        "label_map": lm.name,
        "label_map_sha256": lm.sha256,
        "readout_label_space": lm.readout_space,
        "topk": list(TOPK),
        "preprocessing": transform_signature(transform),
        "eval_version": __version__,
        **run_provenance(transform, resolved),
        **meta,
    }
    return ClassificationResults(summary, predictions)


def run_provenance(transform, device) -> dict:
    device = torch.device(device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    tf32 = bool(torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32) if gpu else False
    return {
        "transform": repr(transform),
        "run_device": str(device),
        "run_gpu": gpu,
        "precision": "tf32" if tf32 else "fp32-strict",
        "run_torch": torch.__version__,
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_user": getpass.getuser(),
        "run_host": platform.node(),
    }
