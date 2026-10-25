"""Phase-2 validation (GPU workstation): our top-1 vs published / recorded numbers, through the real store.

    uv run --no-sync --extra models --with-editable ../model-rearing python scripts/validate_phase2.py [--local]

References:
- torchvision: published IMAGENET1K_* top-1 (torchvision docs), measured on the original val JPEGs.
- model-rearing: the record's probe-phase clean top-1 (`probe/readout/logits_acc`, `probe/<source>_acc`), measured
  during training on the s256 cache through slipstream's DecodeResizeCrop: compare on *-s256, expect ~0.1-0.3 pt
  (different resize filter); originals shown for information.
Writes results/validation_phase2.csv and prints the table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from visionlab.evals.classification.store import ResultsStore

ROOT = Path(__file__).resolve().parents[1]

TORCHVISION = [  # spec, published top-1
    ("pytorch/alexnet:IMAGENET1K_V1", 0.56522),
    ("pytorch/resnet50:IMAGENET1K_V1", 0.76130),
    ("pytorch/resnet50:IMAGENET1K_V2", 0.80858),
    ("pytorch/vit_b_16:IMAGENET1K_V1", 0.81072),
]
P = "core.projector.layers"
REARING = [  # spec, dataset space, recorded probe-phase top-1
    ("2ad57e0f", "in1k", 0.57778),
    (f"2ad57e0f@probe:{P}.fc_block_7", "in1k", 0.56862),
    ("2ad57e0f@probe:core.backbone.avgpool", "in1k", 0.41842),
    ("80a62e22", "in100", 0.7640),
    # a1a37c03 (alexnet2012_out100, 0.6992): pre-v0.5.0 record from another repo, not served by the rearing source
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="local store only (no S3)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=16)
    args = ap.parse_args()
    store = ResultsStore(bucket=None) if args.local else ResultsStore()
    kw = {"batch_size": args.batch_size, "num_workers": args.num_workers, "progress": False}

    jobs = [(spec, "in1k-val", ref, "published (originals)") for spec, ref in TORCHVISION]
    for spec, space, ref in REARING:
        for ds in (f"{space}-val-s256", f"{space}-val"):
            jobs.append((spec, ds, ref, "probe phase (s256)"))

    rows = []
    for spec, dataset, ref, ref_kind in jobs:
        print(f"== {spec} on {dataset}", flush=True)
        r = store.run(spec, dataset=dataset, **kw)
        s = r.summary
        rows.append(
            {
                "spec": spec,
                "model_id": s["model_id"],
                "readout": s["readout_spec"],
                "dataset": dataset,
                "top1": s["top1"],
                "top5": s["top5"],
                "reference": ref,
                "reference_kind": ref_kind,
                "delta_pt": 100 * (s["top1"] - ref),
                "label_map": s["label_map"],
                "run_id": s["run_id"],
                "eval_spec_id": s["eval_spec_id"],
            }
        )
        print(f"   top1 {s['top1']:.5f}  ref {ref:.5f}  delta {100 * (s['top1'] - ref):+.3f} pt", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "validation_phase2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df[["spec", "dataset", "top1", "reference", "delta_pt"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
