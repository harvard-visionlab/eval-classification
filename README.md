# eval-classification

Image classification eval: per-image top-1/top-5, cross-entropy and decision margins
(`(S_target − max_{j≠target} S_j)/√2`, Alvarez & Konkle 2024 UniReps) on visionlab.datasets val sets, for any model
from any [model-evals](https://github.com/harvard-visionlab/model-evals) source (visionlab.models, model-rearing).

```bash
uv sync --extra models
uv run model eval classification pytorch/alexnet:DEFAULT                     # in1k-val (default)
uv run model eval classification <rearing id> --dataset in100-val            # native head
uv run model eval classification <rearing id>@probe:<layer> --dataset in100-val
uv run model query classification
```

```python
from visionlab.evals.classification import classification_eval
results = classification_eval(model, transforms.val, "in1k-val")   # any model + its eval transform, no store
results.summary["top1"], results.predictions                          # per image
```

## Datasets

| name | visionlab.datasets (>= 0.19) | label space |
|---|---|---|
| `in1k-val` (default) | imagenet1k / val / original: the ILSVRC2012 val JPEGs as distributed | in1k (1000) |
| `in100-val` | imagenet100 / val / original | in100 (100) |
| `in1k-val-s256` | imagenet1k / val / jpeg (pre-resized, shortest edge 256, q100) | in1k |
| `in100-val-s256` | imagenet100 / val / jpeg (pre-resized) | in100 |

Originals are what published top-1 numbers use. The s256 caches are faster but differ slightly (alexnet: 0.5638
vs 0.5652 published). Images are decoded with PIL and converted to RGB. The store's content identity
(`ds.identity`, "sha256:…") is hashed into the eval spec.

## Label spaces

A readout scores classes in its own label space: config `label_space`, else its readout metadata, else the source's
`provenance.train_data`, else in1k. The map to the dataset's labels is named and hashed into the spec:
in1k on in100-val keeps the 100 in1k columns of the in100 classes (`in1k->in100:restrict@1`). An in100 readout is
evaluated on in100-val. Override: `model eval classification <spec> --set label_space=in100`.

Preprocessing: the model's `val` transform by default (its standard ImageNet eval pipeline, e.g. Resize 256 +
CenterCrop 224, as published top-1 uses); `--set transform=test` for shortest-edge-to-crop. Both are hashed.

## Results

`results.parquet` per run: index, path, target, pred, correct1, correct5, topk_labels, topk_scores, target_score,
max_nontarget_score, decision_margin, loss (+ identity columns). Summary: top1, top5 (fractions), loss_mean,
dm_mean/median/min/max/kurtosis, n_images. Store: `eval=classification/version=<v>/dataset=…` (model-evals Hive
store).

## Development

```bash
uv run pytest            # unit tests + a 64-image CPU end-to-end on in100-val (marker `data`)
uv run ruff check classification tests
```

History: v0.1 (2024) was a wasabi/notion-backed `ClassifierStore`. Its results were abandoned (George, 2026-10-25).
See PLAN.md.
