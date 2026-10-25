"""Label spaces and the readout → dataset label map.

A readout scores N classes in its training label space; a dataset labels images in its own. The map between them
is explicit, named and versioned, and its content hash goes into the eval spec:

    in1k readout  on in1k-val   identity
    in100 readout on in100-val  identity
    in1k readout  on in100-val  restrict: keep the 100 in1k columns of the in100 classes (in100 order)
    in100 readout on in1k-val   refused: evaluate it on in100-val (the same images, in the readout's space)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

LABEL_SPACES = {"in1k": 1000, "in100": 100}
_ALIASES = {
    "in1k": "in1k",
    "in1000": "in1k",
    "imagenet1k": "in1k",
    "imagenet-1k": "in1k",
    "imagenet": "in1k",
    "in100": "in100",
    "imagenet100": "in100",
    "imagenet-100": "in100",
}


def normalize_space(name: str | None) -> str | None:
    """'imagenet100' -> 'in100'; unknown names pass through unchanged (and are refused by `label_map`)."""
    if name is None:
        return None
    return _ALIASES.get(str(name).lower().split("_s")[0], str(name))


def in100_to_in1000() -> list[int]:
    """in1k index of each in100 class, in in100 order (visionlab.datasets is the single source)."""
    from visionlab.datasets._configs.imagenet_subsets import IN100_TO_IN1000

    return [int(IN100_TO_IN1000[i]) for i in range(100)]


@dataclass(frozen=True)
class LabelMap:
    name: str  # e.g. 'identity', 'in1k->in100:restrict@1'
    readout_space: str
    dataset_space: str
    columns: tuple[int, ...] | None = None  # readout columns kept, in dataset label order; None = identity

    @property
    def sha256(self) -> str:
        blob = json.dumps({"name": self.name, "columns": self.columns}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def signature(self) -> dict:
        return {
            "name": self.name,
            "readout_space": self.readout_space,
            "dataset_space": self.dataset_space,
            "sha256": self.sha256,
        }

    def apply(self, scores):
        """(B, N_readout) -> (B, N_dataset). Works on numpy arrays and torch tensors."""
        n = LABEL_SPACES[self.readout_space]
        if scores.shape[1] != n:
            raise ValueError(
                f"readout gives {scores.shape[1]} scores, label space {self.readout_space!r} has {n}; "
                "set config label_space to the readout's space"
            )
        if self.columns is None:
            return scores
        return scores[:, list(self.columns)] if not isinstance(scores, np.ndarray) else scores[:, self.columns]


def label_map(readout_space: str, dataset_space: str) -> LabelMap:
    if readout_space not in LABEL_SPACES:
        raise ValueError(f"unknown readout label space {readout_space!r}; known: {sorted(LABEL_SPACES)}")
    if readout_space == dataset_space:
        return LabelMap("identity", readout_space, dataset_space)
    if (readout_space, dataset_space) == ("in1k", "in100"):
        return LabelMap("in1k->in100:restrict@1", "in1k", "in100", tuple(in100_to_in1000()))
    if (readout_space, dataset_space) == ("in100", "in1k"):
        raise ValueError("an in100 readout is evaluated on in100-val (the same images, in its own label space)")
    raise ValueError(f"no label map from {readout_space!r} to {dataset_space!r}")
