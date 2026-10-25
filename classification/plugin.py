"""The classification eval as a visionlab.evals plugin: `model eval classification <spec> [--dataset in100-val]`.

Registered under the `visionlab.evals.evals` entry-point group. Config:
- `transform`: 'val' (default: the model's standard ImageNet eval pipeline, e.g. Resize 256 + CenterCrop 224, which
  published top-1 numbers use) or 'test' (shortest edge to the crop size: more field of view)
- `label_space` ('in1k' | 'in100'): overrides the readout's inferred label space (store.readout_label_space)
"""

from __future__ import annotations

from visionlab.evals.core.spec import current_precision

from .data import DATASETS, DEFAULT_DATASET
from .eval import classification_eval
from .store import EVAL_NAME, ResultsStore, cached_dataset
from .version import __version__

CONFIG_KEYS = {"label_space", "transform"}


def pick_transform(transforms, config: dict):
    name = config.get("transform", "val")
    if name not in ("val", "test"):
        raise ValueError(f"classification: transform must be 'val' or 'test', got {name!r}")
    t = getattr(transforms, name)
    if t is None:
        raise ValueError(f"classification: this model source gives no {name!r} transform")
    return t


class ClassificationEval:
    name = EVAL_NAME
    version = __version__
    datasets = tuple(DATASETS)
    default_dataset = DEFAULT_DATASET
    needs = "scores"

    def store(self, **kwargs) -> ResultsStore:
        return ResultsStore(**kwargs)

    def spec(self, identity, transforms, dataset, config, *, seed=None, previous=None):
        unknown = set(config) - CONFIG_KEYS
        if unknown:
            raise ValueError(f"classification: unknown config keys {sorted(unknown)}; allowed: {sorted(CONFIG_KEYS)}")
        return ResultsStore().make_spec(
            identity,
            dataset,
            pick_transform(transforms, config),
            config.get("label_space"),
            seed=seed,
            precision=previous.execution.precision if previous else current_precision(),
            dataset_revision=previous.dataset.revision if previous else "pin",
            transform_name=config.get("transform", "val"),
        )

    def run(self, loaded, spec, **kwargs):
        ds = cached_dataset(spec.dataset.name)
        if ds.revision != spec.dataset.revision:
            raise RuntimeError(
                f"{spec.dataset.name} changed since the spec was built ({ds.revision} != {spec.dataset.revision})"
            )
        if spec.execution.seed is not None:
            kwargs.setdefault("run_seed", spec.execution.seed)
        transform = pick_transform(loaded.transforms, {"transform": spec.transform})
        return classification_eval(loaded.module, transform, ds, label_space=spec.label_map["readout_space"], **kwargs)


EVAL = ClassificationEval()
