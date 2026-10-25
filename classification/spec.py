"""Eval-spec identity for classification (identity contract v2, eval side).

Everything besides the model configuration that determines a result: the data (visionlab.datasets cache + its
content revision), the readout → dataset label map, preprocessing, top-k, numerics. sha256 = `eval_spec_id`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from visionlab.evals.core.spec import canonical_json, sha256_hex

from .scoring import TOPK
from .version import __version__

SPEC_SCHEMA = "visionlab-evals/classification/spec@1"


@dataclass(frozen=True)
class DatasetRef:
    name: str  # eval dataset name (store partition)
    source: str  # visionlab.datasets 'name/split/fmt'
    revision: str | None  # cache content revision (data.cache_revision); None = unknown (not reusable)


@dataclass(frozen=True)
class Execution:
    seed: int | None = None  # random-init models only
    precision: str = "fp32-strict"


@dataclass(frozen=True)
class EvalSpec:
    config_id: str
    dataset: DatasetRef
    label_map: dict  # labels.LabelMap.signature()
    preprocessing: dict  # core.spec.transform_signature
    transform: str = "val"  # which of the model's transforms (the signature above is what is hashed)
    topk: tuple[int, ...] = TOPK
    execution: Execution = field(default_factory=Execution)
    eval_name: str = "classification"
    eval_version: str = __version__
    config_id_source: str = "models-manifest"

    def hashed(self) -> dict:
        return {
            "schema": SPEC_SCHEMA,
            "eval_name": self.eval_name,
            "eval_version": self.eval_version,
            "config_id": self.config_id,
            "dataset": asdict(self.dataset),
            "label_map": self.label_map,
            "preprocessing": {k: v for k, v in self.preprocessing.items() if k != "exact"},
            "topk": list(self.topk),
            "execution": asdict(self.execution),
        }

    @property
    def eval_spec_id(self) -> str:
        return sha256_hex(canonical_json(self.hashed()))

    @property
    def dataset_name(self) -> str:
        return self.dataset.name

    @property
    def dataset_revision(self) -> str | None:
        return self.dataset.revision

    @property
    def reusable(self) -> bool:
        return (
            self.dataset.revision is not None
            and self.preprocessing.get("exact", False)
            and self.config_id_source == "models-manifest"
        )

    def to_dict(self) -> dict:
        return {
            **self.hashed(),
            "transform": self.transform,
            "eval_spec_id": self.eval_spec_id,
            "reusable": self.reusable,
            "preprocessing_exact": self.preprocessing.get("exact", False),
            "config_id_source": self.config_id_source,
        }
