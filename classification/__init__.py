"""visionlab.evals.classification — image classification eval (top-1/top-5 + decision margins).

from visionlab.evals.classification import classification_eval
results = classification_eval(model, transforms.val, "in1k-val")     # any model + its eval transform
model eval classification pytorch/alexnet:DEFAULT --dataset in1k-val  # lab: any model source, stored runs
"""

from .data import DATASETS, DEFAULT_DATASET, EvalDataset, load_eval_dataset
from .eval import classification_eval, predict_scores
from .labels import LABEL_SPACES, LabelMap, label_map
from .scoring import ClassificationResults, build_predictions, summarize
from .version import __version__

__all__ = [
    "DATASETS",
    "DEFAULT_DATASET",
    "LABEL_SPACES",
    "ClassificationResults",
    "EvalDataset",
    "LabelMap",
    "__version__",
    "build_predictions",
    "classification_eval",
    "label_map",
    "load_eval_dataset",
    "predict_scores",
    "summarize",
]
