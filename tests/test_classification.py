"""Scoring, label maps and spec identity (no data); a tiny CPU end-to-end on in100-val (marker `data`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchvision import transforms as T
from visionlab.evals.classification import build_predictions, label_map, summarize
from visionlab.evals.classification.labels import in100_to_in1000, normalize_space
from visionlab.evals.classification.store import ResultsStore, readout_label_space
from visionlab.evals.core.identity import ModelIdentity, ReadoutIdentity

TFM = T.Compose(
    [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)


def test_predictions_topk_margin_loss():
    scores = np.array(
        [
            [3.0, 1.0, 0.0, -1.0, -2.0, -3.0],  # target 0: correct, margin (3-1)/sqrt2
            [0.0, 2.0, 1.0, -1.0, -2.0, -3.0],
        ]
    )  # target 2: top-2 only
    df = build_predictions(scores, np.array([0, 2]), topk=(1, 2))
    assert df["correct1"].tolist() == [True, False] and df["correct2"].tolist() == [True, True]
    assert df["decision_margin"].tolist() == pytest.approx([2 / math.sqrt(2), -1 / math.sqrt(2)])
    assert df["loss"][0] == pytest.approx(-(3 - np.log(np.exp(scores[0]).sum())))
    s = summarize(build_predictions(scores, np.array([0, 2])))
    assert s["top1"] == 0.5 and s["top5"] == 1.0 and s["n_images"] == 2


def test_label_maps():
    assert label_map("in1k", "in1k").columns is None
    lm = label_map("in1k", "in100")
    cols = in100_to_in1000()
    assert len(set(cols)) == 100 and lm.columns == tuple(cols)
    x = np.arange(2000.0).reshape(2, 1000)
    assert lm.apply(x).shape == (2, 100) and lm.apply(x)[0, 0] == cols[0]
    assert torch.equal(lm.apply(torch.tensor(x)), torch.tensor(x[:, cols]))
    with pytest.raises(ValueError, match="in100-val"):
        label_map("in100", "in1k")
    with pytest.raises(ValueError, match="readout gives 100"):
        label_map("in1k", "in1k").apply(np.zeros((1, 100)))
    assert normalize_space("imagenet100") == "in100" and normalize_space("imagenet1k") == "in1k"


def _ident(readout=None, provenance=None):
    return ModelIdentity(
        "rearing",
        "toy",
        "026ede57",
        readout=readout,
        config_id="c" * 64,
        manifest={"configuration": {}, "provenance": provenance or {}},
    )


def test_readout_label_space_inference():
    assert readout_label_space(_ident()) == "in1k"
    assert readout_label_space(_ident(provenance={"train_data": "imagenet100"})) == "in100"
    probe = ReadoutIdentity("probe", "a1b2c3d4", readout_layer="layer4", readout_n_classes=100)
    assert readout_label_space(_ident(probe)) == "in100"
    assert readout_label_space(_ident(probe), override="in1k") == "in1k"


def test_spec_identity(tmp_path):
    store = ResultsStore(bucket=None, cache_dir=tmp_path)
    a = store.make_spec(_ident(), "in1k-val", TFM, dataset_revision="r1")
    assert a.reusable and a.dataset_name == "in1k-val"
    assert store.make_spec(_ident(), "in1k-val", TFM, dataset_revision="r2").eval_spec_id != a.eval_spec_id
    assert (
        store.make_spec(_ident(), "in100-val", TFM, dataset_revision="r1").label_map["name"] == "in1k->in100:restrict@1"
    )
    assert not store.make_spec(_ident(), "in1k-val", TFM, dataset_revision=None).reusable
    assert (
        store.make_spec(_ident(), "in1k-val", TFM, precision="tf32", dataset_revision="r1").eval_spec_id
        != a.eval_spec_id
    )


@pytest.mark.data
def test_e2e_in100_val_subset_cpu(tmp_path):
    """Random-weight resnet18 head on 64 in100-val images: plumbing only (chance-level numbers)."""
    from torchvision.models import resnet18
    from visionlab.evals.classification import classification_eval
    from visionlab.evals.classification.data import EvalDataset, load_eval_dataset

    full = load_eval_dataset("in100-val")
    sub = EvalDataset(full.spec, torch.utils.data.Subset(full.base, range(64)), full.revision)
    torch.manual_seed(0)
    res = classification_eval(
        resnet18(num_classes=1000).eval(), TFM, sub, label_space="in1k", num_workers=0, batch_size=32, progress=False
    )
    assert res.summary["n_images"] == 64 and res.summary["label_map"] == "in1k->in100:restrict@1"
    assert len(res.predictions["topk_labels"][0]) == 5 and res.predictions["path"].notna().all()
    back = type(res).load(res.save(tmp_path / "r"))
    assert back.summary["top1"] == res.summary["top1"]
