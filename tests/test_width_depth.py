from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/split-sharing-20261017"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCH))

import split_model
import run_split


def _same_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_native_width_depth_costs_and_seeded_state_dicts():
    from recurrent_cnn.costs import mac_breakdown, parameter_count
    from recurrent_cnn.models import ModelSpec, build_model

    expected = {
        "C4W32": ("C", 4, 32, 22_724, 19_098_752, 8, 1),
        "B4W45": ("B", 4, 45, 152_335, 37_640_340, 1, 4),
    }
    for arm, (family, depth, width, parameters, macs, groups, branches) in expected.items():
        metadata = split_model.ARM_METADATA[arm]
        spec = ModelSpec(family, depth, width=width)
        model = split_model.build_split_model(spec, seed=817, arm=arm)
        native = build_model(spec, seed=817)
        assert metadata["branch_map"] is None
        assert metadata["init_source_indices"] is None
        assert metadata["parameter_count"] == parameters
        assert metadata["macs"] == macs
        assert parameter_count(model) == parameters
        assert mac_breakdown(spec)["mac_breakdown"]["total"] == macs
        assert len(model.branches) == branches
        assert model.branches[0].norm1.num_groups == groups
        _same_state(model.state_dict(), native.state_dict())


def test_native_factory_passthrough_cost_and_restoration():
    from recurrent_cnn import engine
    from recurrent_cnn.models import ModelSpec, build_model

    original_factory, original_cost = engine.build_model, engine.mac_breakdown
    for arm, family, depth, width, parameters, macs in (
        ("C4W32", "C", 4, 32, 22_724, 19_098_752),
        ("B4W45", "B", 4, 45, 152_335, 37_640_340),
    ):
        spec = ModelSpec(family, depth, width=width)
        with split_model.mapped_engine_factory(engine, arm):
            model = engine.build_model(spec, seed=271)
            _same_state(model.state_dict(), build_model(spec, seed=271).state_dict())
            assert sum(parameter.numel() for parameter in model.parameters()) == parameters
            assert engine.mac_breakdown(spec)["mac_breakdown"]["total"] == macs
        assert engine.build_model is original_factory
        assert engine.mac_breakdown is original_cost


def test_native_initialization_relations_and_paired_config_seeds(tmp_path):
    from recurrent_cnn.models import ModelSpec, build_model

    c4w32 = build_model(ModelSpec("C", 4, width=32), seed=817)
    c8 = build_model(ModelSpec("C", 8, width=32), seed=817)
    for part in ("stem", "head"):
        _same_state(getattr(c4w32, part).state_dict(), getattr(c8, part).state_dict())
    _same_state(c4w32.branches[0].state_dict(), c8.branches[0].state_dict())

    b4w45 = build_model(ModelSpec("B", 4, width=45), seed=817)
    c4w45 = build_model(ModelSpec("C", 4, width=45), seed=817)
    for part in ("stem", "head"):
        _same_state(getattr(b4w45, part).state_dict(), getattr(c4w45, part).state_dict())
    _same_state(b4w45.branches[0].state_dict(), c4w45.branches[0].state_dict())

    args = run_split._parser().parse_args([
        "--data-root", str(tmp_path / "unused-data"), "--output", str(tmp_path / "unused-output"),
        "--synthetic-per-class", "4",
    ])
    configs = {}
    identities = {}
    for arm in ("C4W32", "B4W45", "C4W45"):
        _, subset, validation, config, data_hash, split_hash, subset_hash = run_split._prepared(args, 23, arm)
        configs[arm] = config
        identities[arm] = (data_hash, split_hash, subset_hash, tuple(subset), tuple(validation))
    assert identities["C4W32"] == identities["B4W45"] == identities["C4W45"]
    assert configs["C4W32"].derived_seeds["init"] == configs["B4W45"].derived_seeds["init"]
    assert configs["B4W45"].derived_seeds["init"] == configs["C4W45"].derived_seeds["init"]
    assert (configs["C4W32"].family, configs["C4W32"].depth, configs["C4W32"].width) == ("C", 4, 32)
    assert (configs["B4W45"].family, configs["B4W45"].depth, configs["B4W45"].width) == ("B", 4, 45)


def test_new_arms_are_whitelisted_without_changing_defaults():
    assert run_split.ARMS == ("C4B4", "B4C4")
    assert {"C4W32", "B4W45"} <= set(run_split.ALL_ARMS)
    assert split_model.NATIVE_ARMS == ("C4W32", "B4W45", "C4W45")
