"""Eval datasets, all from visionlab.datasets.

    ds = load_eval_dataset("in1k-val")      # EvalDataset: ds[i] -> (image, label, index, path)
    ds.revision                             # content identity of the store ('sha256:…'; visionlab.datasets)

`in1k-val` / `in100-val` are the original ILSVRC2012 val JPEGs (visionlab.datasets >= 0.19, fmt='original'; the
pipeline published top-1 numbers use). `*-s256` are the pre-resized caches (shortest edge 256, JPEG q100) used for
training-time eval: faster, but top-1 differs slightly from originals. Images are decoded here with PIL and converted
to RGB (ImageNet val has grayscale and CMYK JPEGs); the model's own transform does the rest.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class DatasetSpec:
    name: str  # eval dataset name: the store's dataset= partition
    source: str  # visionlab.datasets name
    split: str
    fmt: str
    label_field: str
    label_space: str


DATASETS = {
    "in1k-val": DatasetSpec("in1k-val", "imagenet1k", "val", "original", "label", "in1k"),
    "in100-val": DatasetSpec("in100-val", "imagenet100", "val", "original", "label", "in100"),
    "in1k-val-s256": DatasetSpec("in1k-val-s256", "imagenet1k", "val", "jpeg", "label", "in1k"),
    "in100-val-s256": DatasetSpec("in100-val-s256", "imagenet100", "val", "jpeg", "label", "in100"),
}
DEFAULT_DATASET = "in1k-val"


def decode(image) -> Image.Image:
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = Image.open(io.BytesIO(bytes(image)))
    return image if image.mode == "RGB" else image.convert("RGB")


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
        image = decode(s["image"])
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

    base = load(spec.source, split=spec.split, fmt=spec.fmt)  # image bytes; decoded in __getitem__
    revision = getattr(base, "identity", None)
    if revision is None:
        warnings.warn(
            f"{spec.source}/{spec.split}/{spec.fmt} has no content identity (no per-file hashes in its manifest), so "
            f"its runs are not reusable; refresh it with `visionlab-datasets sync {spec.source} {spec.split} "
            f"{spec.fmt} --deep`",
            stacklevel=2,
        )
    return EvalDataset(spec, base, revision, transform)
