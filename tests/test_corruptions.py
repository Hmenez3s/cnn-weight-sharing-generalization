from __future__ import annotations

import numpy as np
import pytest

from recurrent_cnn.data import (
    CIFAR100C_CORRUPTIONS,
    _open_cifar100_c,
    make_split,
    make_validation_corruption,
    prepare_data,
)
from recurrent_cnn.config import TrainConfig
from recurrent_cnn.engine import train_run


def test_validation_transforms_are_repeatable_bound_to_canonical_holdout() -> None:
    arrays = prepare_data("unused", synthetic=True, synthetic_per_class=12, seed=3)
    validation = make_split(arrays.labels, 17, synthetic=True)[1]
    for name in ("gaussian_blur", "saturate", "spatter"):
        first = make_validation_corruption(arrays, validation, name, 3, 99, split_seed=17)
        again = make_validation_corruption(arrays, validation, name, 3, 99, split_seed=17)
        assert first.namespace == "validation_corruption"
        assert first.metadata["source"] == "heldout_validation"
        assert first.metadata["origin_namespace"] == "synthetic"
        assert first.metadata["transform"] == name
        assert first.metadata["severity"] == 3 and first.metadata["seed"] == 99
        assert first.metadata["evaluation_namespace"] == f"validation-corruption-{name}-s3"
        assert first.metadata["identity"] == again.metadata["identity"]
        assert np.array_equal(first.images, again.images)
        assert np.array_equal(first.labels, arrays.labels[validation])
        assert first.images.dtype == np.uint8
    with pytest.raises(ValueError, match="canonical validation"):
        make_validation_corruption(arrays, make_split(arrays.labels, 17, synthetic=True)[0][:5], "spatter", 1, 0, split_seed=17)
    with pytest.raises(ValueError, match="unknown validation corruption"):
        make_validation_corruption(arrays, validation, "gaussian_noise", 1, 0, split_seed=17)
    with pytest.raises(ValueError, match="severity"):
        make_validation_corruption(arrays, validation, "spatter", True, 0, split_seed=17)


def test_cifar100c_reader_aligns_exact_readonly_severity_slice(tmp_path) -> None:
    path = tmp_path / "gaussian_noise.npy"
    images = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(50_000, 32, 32, 3))
    # The first four bytes encode the global sample index, an independent
    # fixture oracle for severity-block offsets and their boundaries.
    for index in (0, 9_999, 10_000, 19_999, 20_000, 29_999, 30_000, 39_999, 40_000, 49_999):
        images[index, 0, 0] = np.frombuffer(index.to_bytes(3, "little"), dtype=np.uint8)
    images.flush()
    del images
    labels = np.tile(np.arange(10_000, dtype=np.int64) % 100, 5)
    np.save(tmp_path / "labels.npy", labels, allow_pickle=False)

    result = _open_cifar100_c(tmp_path, "gaussian_noise", 3)
    assert result.images.shape == (10_000, 32, 32, 3)
    assert result.images.dtype == np.uint8 and result.labels.dtype == np.int64
    assert not result.images.flags.writeable and not result.labels.flags.writeable
    assert result.images[0, 0, 0].tobytes() == (20_000).to_bytes(3, "little")
    assert result.images[-1, 0, 0].tobytes() == (29_999).to_bytes(3, "little")
    assert np.array_equal(result.labels[[0, -1]], labels[[20_000, 29_999]])
    assert result.metadata == {
        "source": "cifar100c", "origin_namespace": "official_test_corruption",
        "corruption": "gaussian_noise", "severity": 3,
        "evaluation_namespace": "cifar100c-gaussian_noise-s3",
        "image_offset": 20_000, "count": 10_000,
    }
    with pytest.raises(ValueError):
        result.images[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="requires arrays namespace 'synthetic'"):
        train_run(TrainConfig("B", 2), result, np.array([0]), np.array([1]), tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
    assert len(CIFAR100C_CORRUPTIONS) == 15


@pytest.mark.parametrize("name,severity", [("unknown", 1), ("gaussian_noise", 0), ("gaussian_noise", 6), ("gaussian_noise", True)])
def test_cifar100c_reader_rejects_unknown_condition_before_open(tmp_path, name, severity) -> None:
    with pytest.raises(ValueError):
        _open_cifar100_c(tmp_path, name, severity)


def test_cifar100c_reader_rejects_malformed_images_and_labels(tmp_path) -> None:
    np.save(tmp_path / "gaussian_noise.npy", np.zeros((12, 32, 32, 3), dtype=np.uint8), allow_pickle=False)
    np.save(tmp_path / "labels.npy", np.zeros(12, dtype=np.int64), allow_pickle=False)
    with pytest.raises(ValueError, match="images"):
        _open_cifar100_c(tmp_path, "gaussian_noise", 1)

    (tmp_path / "gaussian_noise.npy").unlink()
    images = np.lib.format.open_memmap(tmp_path / "gaussian_noise.npy", mode="w+", dtype=np.uint16,
                                      shape=(50_000, 32, 32, 3))
    images.flush()
    del images
    np.save(tmp_path / "labels.npy", np.zeros(50_000, dtype=np.int64), allow_pickle=False)
    with pytest.raises(ValueError, match="images"):
        _open_cifar100_c(tmp_path, "gaussian_noise", 1)

    (tmp_path / "gaussian_noise.npy").unlink()
    images = np.lib.format.open_memmap(tmp_path / "gaussian_noise.npy", mode="w+", dtype=np.uint8,
                                      shape=(50_000, 32, 32, 3))
    images.flush()
    del images
    np.save(tmp_path / "labels.npy", np.zeros(50_000, dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="labels"):
        _open_cifar100_c(tmp_path, "gaussian_noise", 1)
    np.save(tmp_path / "labels.npy", np.full(50_000, 100, dtype=np.int64), allow_pickle=False)
    with pytest.raises(ValueError, match=r"\[0, 99\]"):
        _open_cifar100_c(tmp_path, "gaussian_noise", 1)
    bad_blocks = np.zeros(50_000, dtype=np.int64)
    bad_blocks[10_000:] = 1
    np.save(tmp_path / "labels.npy", bad_blocks, allow_pickle=False)
    with pytest.raises(ValueError, match="repeat exactly"):
        _open_cifar100_c(tmp_path, "gaussian_noise", 1)
    np.save(tmp_path / "labels.npy", np.zeros(50_000, dtype=np.int64), allow_pickle=False)
    with pytest.raises(ValueError, match="100 examples per class"):
        _open_cifar100_c(tmp_path, "gaussian_noise", 1)


def test_private_reader_only_opens_requested_corruption(tmp_path, monkeypatch) -> None:
    real_load = np.load
    opened = []

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        assert kwargs.get("allow_pickle") is False
        assert kwargs.get("mmap_mode") == "r"
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy)
    with pytest.raises(ValueError):
        _open_cifar100_c(tmp_path, "not-a-condition", 1)
    assert opened == []
