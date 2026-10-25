# eval-classification revival — plan (approved 2026-10-25)

## Today (last commit 2024-11-21)
- `validation(model, dataset, layer_names, map_output)`: per-image top-1/top-5, loss, decision margin
  (UniReps eq. 4), summary (top1, top5, dm stats). The core logic is sound and short (~150 lines).
- `ClassifierStore` (personal_hub.py): wasabi `visionlab-results/<user>/dnn-evals`, s3_filestore, notion, a
  `model.id` monkey-patch, old `visionlab_models`/`visionlab_datasets` APIs. All of this is superseded.
- `knn.py` and `linear_probe.py` are 2-line stubs. `setup.py`, no tests.

## Goal
`model eval classification <any spec> --dataset <set>` for models-redux and model-rearing models (any readout), with
results in the Hive store keyed like anagrams. Plus a standalone `classification_eval(model, transform, dataset)`
that needs no lab infra.

## Design
**Package**: `visionlab.evals.classification` (src layout, pyproject + uv, same torch indexes as eval-anagrams), with
`lab` and `models` extras like eval-anagrams. Keep the git history. Plugin `classification` in `visionlab.evals.evals`.

**Readouts replace `layer_names`**: the old "probe = a layer whose output is logits" mode becomes the source's
readout (`026ede57@probe:layer4.1`, `facebook/dinov2_vitb14:DEFAULT@lc4`). Each readout is its own run at its own
Hive address.

**Label space** (the one real design problem): a readout scores N classes in its training label space
(`readout_n_classes`, `readout_train_data`: in1k, in100, ...). A dataset has its own label space. The eval maps
readout → dataset explicitly:
- same space: identity
- in100 model on in1k-val: score only the 100 classes' images, `in100_to_in1000` from visionlab.datasets
- in1k model on in100-val: restrict to the 100 columns
- otherwise: refuse
The mapping is named and versioned (`label_map: "in100_to_in1000@1"`) and hashed into the eval spec.

**Eval spec** (hashed): dataset name + pinned revision/shard hash, label map, preprocessing (`transforms.test`
signature), topk, precision, eval version.

**Per-image results**: index, path, target, pred, correct1/5, top5 labels and scores, target score, max non-target
score, decision_margin, loss. **Summary**: top1, top5, dm_mean/median/min/max/kurtosis, n.

**Data**: visionlab.datasets streaming val sets, decoded with PIL, then the model's own `transforms.test`.

## Phases
1. Repo modernization + `classification_eval` + plugin + tests (synthetic + tiny CPU e2e) on in1k-val.
2. Validation on the workstation (GPU):
   - torchvision published top-1, e.g. alexnet 56.52 and resnet50 V1 76.13; tolerance ±0.1 depending on data source
   - a model-rearing model's `@logits` and `@probe:` top-1 vs its own probe-phase numbers (rearing uses a 256/224
     pipeline, so expect ~1 px field-of-view differences)
   - this is also the "in1k-verified" check models-redux's HOUSE_RULES ask for
3. More sets (in100-val, imagenette, robustness sets) as each becomes available in visionlab.datasets.
4. (Later, separate decision) kNN and linear-probe *training*: these produce readouts, not scores. A trained probe
   should become a readout artifact (models-redux head-file format) so `@probe:` works everywhere, rather than living
   inside this eval.

## Decisions (George, 2026-10-25)
1. **Data comes from visionlab.datasets, whichever images we use.** It has no originals today (in1k and in100 are
   only `s256_l512_q100` caches). The datasets session was asked to plan adding original val images (in1k and in100)
   and to verify that plan with George before building. Phase 1 develops against the existing caches. Its validation
   runs on originals once they exist, and the data source is part of the eval spec (dataset name + cache identity).
2. Phase-1 datasets: **in1k-val and in100-val**.
3. Old wasabi results: **abandon**.
4. kNN / linear probes: **deferred** to phase 4.

## Status (2026-10-25)
Phase 1 code done: `classification_eval`, label maps, spec, store, `model eval classification` plugin; 5 tests pass.
Smoke runs (s256 caches; laptop CPU, which was a mistake: full sets go to vesper from now on):
- torchvision alexnet, in1k-val: `val` transform top-1 0.56384 vs published 0.56522 (−0.14 pt, expected from the
  pre-resized/re-encoded cache); `test` transform 0.54358 (so the default is `val`)
- rearing/alexnet2023_w1:2ad57e0f head, in1k-val: 0.57822 vs its probe-phase clean top-1 0.57778
- same model `@probe:core.projector.layers.fc_block_7` on in100-val (in1k->in100 restrict): 0.7882
Waiting on: visionlab.datasets originals (plan with George). Then phase 2 validation on vesper.

## Phase 2 validation (vesper RTX 4090, 2026-10-25; `scripts/validate_phase2.py`, runs stored in s3://visionlab-evals)

| spec | dataset | top1 | reference | Δ pt |
|---|---|---|---|---|
| pytorch/alexnet:IMAGENET1K_V1 | in1k-val | 0.56556 | 0.56522 published | +0.034 |
| pytorch/resnet50:IMAGENET1K_V1 | in1k-val | 0.76146 | 0.76130 published | +0.016 |
| pytorch/resnet50:IMAGENET1K_V2 | in1k-val | 0.80842 | 0.80858 published | −0.016 |
| pytorch/vit_b_16:IMAGENET1K_V1 | in1k-val | 0.81066 | 0.81072 published | −0.006 |
| rearing 2ad57e0f (head) | in1k-val-s256 | 0.57820 | 0.57778 probe phase | +0.042 |
| rearing 2ad57e0f (head) | in1k-val | 0.57562 | | −0.216 |
| 2ad57e0f@probe:…fc_block_7 | in1k-val-s256 | 0.56910 | 0.56862 | +0.048 |
| 2ad57e0f@probe:…fc_block_7 | in1k-val | 0.56760 | | −0.102 |
| 2ad57e0f@probe:core.backbone.avgpool | in1k-val-s256 | 0.41744 | 0.41842 | −0.098 |
| 2ad57e0f@probe:core.backbone.avgpool | in1k-val | 0.41582 | | −0.260 |
| rearing 80a62e22 (in100 head) | in100-val-s256 | 0.76380 | 0.76400 | −0.020 |
| rearing 80a62e22 (in100 head) | in100-val | 0.76240 | | −0.160 |

- Published torchvision numbers are reproduced on originals within ±0.035 pt (datasets' suggested tolerance: ±0.05).
- Rearing models match their probe-phase numbers on the cache they were evaluated on (s256) within ±0.1 pt. That
  probe phase used slipstream's resize, so it isn't the identical pipeline. On originals they score 0.1–0.26 pt
  lower, consistently: these models were trained and probed on s256 images.
- Fixes this found: DataLoader workers on slipstream stores (slipstream fixed it in 0.11.3; we use a thread pool
  anyway), CPU-by-default device (eval and model-evals run_eval now default to CUDA, so probes move with the model).
