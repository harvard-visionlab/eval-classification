"""Classification results store: the model-evals Hive store bound to this eval, plus spec construction.

store = ResultsStore()
store.run("pytorch/alexnet:DEFAULT", dataset="in1k-val")   # resolve, eval, save (or return the matching run)
store.query(dataset="in1k-val")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from visionlab.evals.core.identity import ModelIdentity
from visionlab.evals.core.spec import current_precision, transform_signature
from visionlab.evals.core.store import DEFAULT_BUCKET, DEFAULT_CACHE
from visionlab.evals.core.store import ResultsStore as _CoreStore

from .data import DATASETS, DEFAULT_DATASET, load_eval_dataset
from .labels import LABEL_SPACES, label_map, normalize_space
from .scoring import ClassificationResults
from .spec import DatasetRef, EvalSpec, Execution
from .version import __version__

EVAL_NAME = "classification"
_N_TO_SPACE = {n: s for s, n in LABEL_SPACES.items()}


@lru_cache(maxsize=4)
def cached_dataset(name: str):
    return load_eval_dataset(name)


def readout_label_space(identity: ModelIdentity, override: str | None = None) -> str:
    """The readout's label space: config override > readout train_data > readout n_classes > the source's
    provenance `train_data` > 'in1k' (native ImageNet heads). A wrong guess fails loudly at run time (score width)."""
    if override:
        return normalize_space(override)
    ro = identity.readout
    if ro is not None and ro.readout_train_data:
        return normalize_space(ro.readout_train_data)
    if ro is not None and ro.readout_n_classes in _N_TO_SPACE:
        return _N_TO_SPACE[ro.readout_n_classes]
    hint = ((identity.manifest or {}).get("provenance") or {}).get("train_data")
    return normalize_space(hint) if hint else "in1k"


class ResultsStore(_CoreStore):
    def __init__(
        self,
        bucket: str | None = DEFAULT_BUCKET,
        cache_dir: str | Path = DEFAULT_CACHE,
        eval_name: str = EVAL_NAME,
        eval_version: str = __version__,
    ):
        super().__init__(eval_name, eval_version, ClassificationResults, bucket, cache_dir, primary_metric="top1")

    def make_spec(
        self,
        identity: ModelIdentity,
        dataset: str,
        transform,
        label_space: str | None = None,
        seed: int | None = None,
        precision: str = "fp32-strict",
        dataset_revision: str | None = "pin",
        transform_name: str = "val",
    ) -> EvalSpec:
        """Build the eval spec before loading the model (cache check). 'pin' reads the cache's content revision."""
        ds = DATASETS[dataset]
        if dataset_revision == "pin":
            dataset_revision = cached_dataset(dataset).revision
        config_id, source = identity.resolved_config_id()
        lm = label_map(readout_label_space(identity, label_space), ds.label_space)
        return EvalSpec(
            config_id=config_id,
            config_id_source=source,
            dataset=DatasetRef(ds.name, f"{ds.source}/{ds.split}/{ds.fmt}", dataset_revision),
            label_map=lm.signature(),
            preprocessing=transform_signature(transform),
            transform=transform_name,
            execution=Execution(seed=seed, precision=precision),
            eval_name=self.eval_name,
            eval_version=self.eval_version,
        )

    def run(
        self,
        spec: str,
        dataset: str = DEFAULT_DATASET,
        force: bool = False,
        seed: int | None = None,
        label_space: str | None = None,
        transform: str = "val",
        **run_kwargs,
    ) -> ClassificationResults:
        """Evaluate a model spec from any installed source and store a run (or return the matching stored run)."""
        from visionlab.evals.core.evals import run_eval

        from .plugin import EVAL

        config = {"transform": transform, **({"label_space": label_space} if label_space else {})}
        return run_eval(EVAL, spec, dataset, config, force=force, seed=seed, store=self, **run_kwargs)


__all__ = ["EVAL_NAME", "ResultsStore", "current_precision", "readout_label_space"]
