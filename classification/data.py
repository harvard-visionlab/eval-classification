"""Eval datasets, all from visionlab.datasets.

    ds = load_eval_dataset("in1k-val")      # EvalDataset: ds[i] -> (image, label, index, path)
    ds.revision                             # content identity of the cache (None if its manifest has no hashes)

Each eval dataset names a visionlab.datasets cache (name, split, fmt), the label field and its label space. When
visionlab.datasets adds original (unresized) val images, they become new entries here.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str  # eval dataset name: the store's dataset= partition
    source: str  # visionlab.datasets name
    split: str
    fmt: str
    label_field: str
    label_space: str


DATASETS = {
    "in1k-val": DatasetSpec("in1k-val", "imagenet1k", "val", "jpeg", "label", "in1k"),
    "in100-val": DatasetSpec("in100-val", "imagenet100", "val", "jpeg", "label", "in100"),
}
DEFAULT_DATASET = "in1k-val"


def cache_revision(local_dir: str | Path) -> str | None:
    """sha256 over the cache's published per-file sha256s (slipstream manifest `file_sha256`); None when the
    manifest has none (an older download: re-sync with visionlab-datasets), which makes runs non-reusable."""
    manifest = json.loads((Path(local_dir) / "manifest.json").read_text())
    files = manifest.get("file_sha256")
    if not files:
        return None
    blob = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class EvalDataset:
    """(image, label, index, path) in dataset order; image is RGB PIL, or `transform(image)` when set."""

    def __init__(self, spec: DatasetSpec, base, revision: str | None, transform=None):
        self.spec = spec
        self.base = base
        self.revision = revision
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        image = s["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(s[self.spec.label_field]), i, s["path"]

    def with_transform(self, transform) -> EvalDataset:
        return EvalDataset(self.spec, self.base, self.revision, transform)


def load_eval_dataset(name: str = DEFAULT_DATASET, transform=None) -> EvalDataset:
    if name not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}, got {name!r}")
    spec = DATASETS[name]
    from visionlab.datasets import load

    base = load(spec.source, split=spec.split, fmt=spec.fmt, decode_images=True)
    revision = cache_revision(base.cache_path)
    if revision is None:
        warnings.warn(
            f"{base.cache_path} has no per-file hashes in its manifest, so its runs are not reusable; refresh it with "
            f"`visionlab-datasets sync {spec.source} {spec.split} {spec.fmt} --deep`",
            stacklevel=2,
        )
    return EvalDataset(spec, base, revision, transform)
