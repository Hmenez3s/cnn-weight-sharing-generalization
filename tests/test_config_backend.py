import pytest

from recurrent_cnn.backend import resolve_backend
from recurrent_cnn.config import TrainConfig, derive_seeds


def test_config_is_strict_canonical_and_seeded():
    cfg = TrainConfig("B", 2, paired_seed=7)
    assert cfg.canonical_json() == TrainConfig.from_mapping(cfg.to_dict()).canonical_json()
    assert cfg.config_hash == TrainConfig.from_yaml(cfg.canonical_json()).config_hash
    assert set(cfg.derived_seeds) == {"subset", "init", "minibatch", "augmentation"}
    assert cfg.derived_seeds == derive_seeds(7)
    assert cfg.derived_seeds["subset"] == cfg.paired_seed
    assert len(set(cfg.derived_seeds.values())) == 4
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_mapping({**cfg.to_dict(), "typo": 1})
    with pytest.raises(ValueError):
        TrainConfig("B", 3)


@pytest.mark.parametrize("field", ["learning_rate", "momentum", "weight_decay"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_numeric_optimizer_config_requires_finite_non_boolean_values(field, value, tmp_path):
    path = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError):
        TrainConfig("B", 2, **{field: value})
    assert not path.exists()


@pytest.mark.parametrize("field,value", [("fraction", float("nan")), ("fraction", float("inf")),
                                          ("fraction", True), ("alpha", float("nan")),
                                          ("alpha", float("inf"))])
def test_fraction_and_alpha_must_be_finite(field, value):
    with pytest.raises(ValueError):
        TrainConfig("B", 2, **{field: value})


def test_backend_never_falls_back_and_rejects_unsupported_amp():
    assert str(resolve_backend("cpu").device) == "cpu"
    with pytest.raises(ValueError, match="until CUDA"):
        resolve_backend("cpu", amp=True)
    with pytest.raises(ValueError, match="device"):
        resolve_backend("auto")
