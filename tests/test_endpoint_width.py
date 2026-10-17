from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/split-sharing-20261017"))


def test_native_c4w45_cost_and_group_norm_metadata():
    from recurrent_cnn.models import ModelSpec, build_model
    from recurrent_cnn.costs import mac_breakdown, parameter_count
    import split_model

    spec = ModelSpec("C", 4, width=45)
    model = build_model(spec, seed=123)
    assert parameter_count(model) == 42_445
    assert mac_breakdown(spec)["mac_breakdown"]["total"] == 37_640_340
    assert split_model.ARM_METADATA["C4W45"]["target_macs"] == 37_973_120
    assert split_model.ARM_METADATA["C4W45"]["group_norm_groups"] == 1
    target = 37_973_120
    candidates = {width: mac_breakdown(ModelSpec("C", 4, width=width))["mac_breakdown"]["total"]
                  for width in range(1, 129)}
    best_width = min(candidates, key=lambda width: (abs(candidates[width] - target), width))
    assert best_width == 45
    mismatch = (candidates[45] - target) / target
    assert abs(mismatch - split_model.ARM_METADATA["C4W45"]["mac_relative_mismatch"]) < 1e-10
    assert model.branches[0].norm1.num_groups == 1
