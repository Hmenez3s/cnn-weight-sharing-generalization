import pytest

from recurrent_cnn.config import TrainConfig


def test_overfit_role_is_explicitly_synthetic_only():
    config = TrainConfig("B", 2, role="overfit", synthetic=True)
    assert config.role == "overfit" and config.synthetic is True
    with pytest.raises(ValueError, match="overfit role requires synthetic=True"):
        TrainConfig("B", 2, role="overfit", synthetic=False)


def test_amended_recipe_is_conditional_and_validates_named_ablation_flags():
    e = TrainConfig("E", 8)
    assert e.to_dict()["recall"] is True
    assert e.to_dict()["variable_depth"] is True
    assert e.to_dict()["progressive_loss"] is True
    assert e.to_dict()["selection_identity"] == "amended_v2"
    f = TrainConfig("F", 8, recall=False, variable_depth=True, progressive_loss=True)
    assert f.to_dict()["recall"] is False
    assert TrainConfig.from_mapping(f.to_dict()).config_hash == f.config_hash
    with pytest.raises(ValueError, match="explicit boolean"):
        TrainConfig("F", 8)
    with pytest.raises(ValueError, match="all mechanisms"):
        TrainConfig("E", 8, progressive_loss=False)
    with pytest.raises(ValueError, match="positive"):
        TrainConfig("E", 8, alpha=0)
    rescue = TrainConfig("E", 8, alpha=0.05, progressive_weight=0.75)
    assert rescue.to_dict()["progressive_weight"] == 0.75
    assert TrainConfig.from_mapping(rescue.to_dict()).config_hash == rescue.config_hash
    for field in ("recall", "variable_depth", "progressive_loss"):
        for invalid in ("yes", 1):
            with pytest.raises(ValueError, match="booleans"):
                TrainConfig("E", 8, **{field: invalid})
    for invalid in (True, float("nan"), -0.01, 1.01):
        with pytest.raises(ValueError, match="progressive_weight"):
            TrainConfig("E", 8, progressive_weight=invalid)
    assert set(TrainConfig("C", 8).derived_seeds) == {"subset", "init", "minibatch", "augmentation"}
    assert "recurrence_path" in e.derived_seeds


def test_amended_control_selection_identity_is_explicit_and_conditional():
    legacy = TrainConfig("B", 4)
    amended = TrainConfig("B", 4, selection_identity="amended_v2")
    assert "selection_identity" not in legacy.to_dict()
    assert amended.to_dict()["selection_identity"] == "amended_v2"
    assert legacy.config_hash != amended.config_hash


def test_existing_default_config_mapping_and_hash_are_unchanged():
    positional = TrainConfig("B", 2, 32, 0.1, 0.5)
    assert positional.fraction == 0.5
    assert positional.recall is None
    config = TrainConfig("B", 2)
    assert config.to_dict() == {
        "family": "B", "depth": 2, "width": 32, "alpha": 0.1, "fraction": 1.0,
        "paired_seed": 0, "data_hash": "", "split_hash": "", "subset_hash": "",
        "role": "smoke", "synthetic": True, "total_updates": 2000, "batch_size": 128,
        "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0005,
        "cosine_horizon": 2000, "validation_cadence": 100, "device": "cpu",
        "amp": False, "num_classes": 100, "split_seed": 0,
    }
    assert config.config_hash == "7fb20a1d504acd1b4fe252afeb1a681de32f19d065bc84106465198f2c7ca3e9"
