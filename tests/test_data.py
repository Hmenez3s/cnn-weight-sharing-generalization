from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from recurrent_cnn.data import (
    CifarArrays,
    FRACTIONS,
    build_manifest,
    canonical_sha256,
    load_manifest,
    make_split,
    make_subsets,
    namespace_id,
    prepare_data,
    save_manifest,
    validate_manifest,
    make_validation_corruption,
)
from recurrent_cnn.config import TrainConfig
from recurrent_cnn.engine import train_run


def official_labels() -> np.ndarray:
    return np.repeat(np.arange(100, dtype=np.int64), 500)


def test_official_split_and_nested_seeded_subsets() -> None:
    labels = official_labels()
    train, validation = make_split(labels, split_seed=417)
    assert len(train) == 45_000
    assert len(validation) == 5_000
    assert not set(train).intersection(validation)
    assert np.all(np.bincount(labels[train], minlength=100) == 450)
    assert np.all(np.bincount(labels[validation], minlength=100) == 50)

    repeated_train, repeated_validation = make_split(labels, split_seed=417)
    assert np.array_equal(train, repeated_train)
    assert np.array_equal(validation, repeated_validation)
    assert not np.array_equal(validation, make_split(labels, 418)[1])

    subsets = make_subsets(labels, train, paired_seed=23)
    assert [len(subsets[f]) for f in FRACTIONS] == [4_500, 11_250, 22_500, 45_000]
    class_counts = np.bincount(labels[subsets[0.25]], minlength=100)
    assert np.count_nonzero(class_counts == 113) == 50
    assert np.count_nonzero(class_counts == 112) == 50
    for fraction, smaller in zip(FRACTIONS[1:], FRACTIONS[:-1]):
        assert set(subsets[smaller]).issubset(subsets[fraction])
    assert all(set(indices).issubset(set(train)) for indices in subsets.values())
    assert all(np.array_equal(subsets[f], make_subsets(labels, train, 23)[f]) for f in FRACTIONS)
    other_seed_subsets = make_subsets(labels, train, 24)
    assert all(not np.array_equal(subsets[f], other_seed_subsets[f]) for f in FRACTIONS[:-1])
    assert np.array_equal(subsets[1.0], other_seed_subsets[1.0])


def test_official_count_contract_is_not_silently_relaxed() -> None:
    labels = np.repeat(np.arange(100, dtype=np.int64), 10)
    with pytest.raises(ValueError, match="official split"):
        make_split(labels, 1)
    train, validation = make_split(labels, 1, synthetic=True)
    assert len(train) + len(validation) == len(labels)
    with pytest.raises(ValueError, match="official subsets"):
        make_subsets(labels, train, 1)
    subsets = make_subsets(labels, train, 1, synthetic=True)
    assert all(len(subsets[a]) <= len(subsets[b]) for a, b in zip(FRACTIONS, FRACTIONS[1:]))
    manifest = build_manifest(
        split_seed=1, paired_seed=1, train_indices=train, validation_indices=validation,
        subsets=subsets, synthetic=True, train_namespace="synthetic",
    )
    assert manifest["synthetic"] is True
    assert manifest["namespaces"] == {"train": "synthetic", "validation": "synthetic"}


def test_manifest_is_canonical_namespaced_and_immutable(tmp_path) -> None:
    labels = official_labels()
    train, validation = make_split(labels, 9)
    subsets = make_subsets(labels, train, 3)
    manifest = build_manifest(
        split_seed=9, paired_seed=3, train_indices=train, validation_indices=validation,
        subsets=subsets,
    )
    repeated = build_manifest(
        split_seed=9, paired_seed=3, train_indices=train.copy(), validation_indices=validation.copy(),
        subsets=subsets,
    )
    assert manifest == repeated
    assert manifest["sha256"] == canonical_sha256({k: v for k, v in manifest.items() if k != "sha256"})
    assert manifest["counts"] == {
        "train": 45_000, "validation": 5_000,
        "subsets": {"0.10": 4_500, "0.25": 11_250, "0.50": 22_500, "1.00": 45_000},
    }
    assert manifest["namespaces"] == {"train": "train", "validation": "train"}
    validation_example = int(validation[0])
    assert namespace_id(manifest["namespaces"]["validation"], validation_example) == f"train:{validation_example}"
    assert namespace_id("train", 2) == "train:2"
    assert namespace_id("test", 2) == "test:2"
    assert namespace_id("train", 2) != namespace_id("test", 2)

    path = save_manifest(tmp_path / "identity.json", manifest)
    assert load_manifest(path) == manifest
    with pytest.raises(FileExistsError):
        save_manifest(path, manifest)
    damaged = json.loads(path.read_text())
    damaged["indices"]["validation"][0] += 1
    with pytest.raises(ValueError, match="SHA256"):
        validate_manifest(damaged)


def test_manifest_rejects_overlap_and_non_nested_subsets() -> None:
    train = np.arange(100, dtype=np.int64)
    validation = np.arange(100, 110, dtype=np.int64)
    subsets = {0.10: train[:10], 0.25: train[:25], 0.50: train[:50], 1.0: train}
    with pytest.raises(ValueError, match="disjoint"):
        build_manifest(split_seed=1, paired_seed=1, train_indices=train, validation_indices=train[:1], subsets=subsets, synthetic=True, train_namespace="synthetic")
    subsets[0.50] = np.arange(50, 150)
    with pytest.raises(ValueError, match="contained"):
        build_manifest(split_seed=1, paired_seed=1, train_indices=train, validation_indices=validation, subsets=subsets, synthetic=True, train_namespace="synthetic")


def test_prepare_data_is_train_only_and_does_not_download_by_default(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []
    integrity_reads: list[str] = []

    class FakeCIFAR100:
        train_list = [("train", "train-checksum")]
        test_list = [("test", "closed-test-checksum")]

        def __init__(self, *, root: str, train: bool, download: bool):
            calls.append({"root": root, "train": train, "download": download})
            # Match torchvision's integrity check even when train=True.
            for filename, _ in self.train_list + self.test_list:
                if filename == "test":
                    raise AssertionError("training preparation must not open test for checksums")
                integrity_reads.append(filename)
            self.data = np.broadcast_to(np.zeros((1, 32, 32, 3), dtype=np.uint8), (50_000, 32, 32, 3))
            self.targets = np.repeat(np.arange(100, dtype=np.int64), 500).tolist()

    monkeypatch.setitem(sys.modules, "torchvision", types.ModuleType("torchvision"))
    datasets = types.ModuleType("torchvision.datasets")
    datasets.CIFAR100 = FakeCIFAR100
    monkeypatch.setitem(sys.modules, "torchvision.datasets", datasets)

    arrays = prepare_data(tmp_path)
    assert calls == [{"root": str(tmp_path), "train": True, "download": False}]
    assert integrity_reads == ["train"]
    assert FakeCIFAR100.test_list == [("test", "closed-test-checksum")]
    assert arrays.namespace == "train"
    assert arrays.images.shape == (50_000, 32, 32, 3)
    assert arrays.labels.dtype == np.int64


def test_eval_only_arrays_are_rejected_before_training_output(tmp_path) -> None:
    arrays = prepare_data(tmp_path / "unused", synthetic=True, synthetic_per_class=12, seed=4)
    validation = make_split(arrays.labels, 7, synthetic=True)[1]
    eval_arrays = make_validation_corruption(arrays, validation, "saturate", 1, 5, split_seed=7)
    with pytest.raises(ValueError, match="requires arrays namespace 'synthetic'"):
        train_run(TrainConfig("B", 2, split_seed=7), eval_arrays, np.array([0]), np.array([1]), tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_synthetic_mode_is_explicit_marked_and_never_loads_torchvision(monkeypatch, tmp_path) -> None:
    def forbidden_import(name, *args, **kwargs):
        if name.startswith("torchvision"):
            raise AssertionError("synthetic mode must not import torchvision")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", forbidden_import)
    arrays = prepare_data(tmp_path / "missing", synthetic=True, synthetic_per_class=5, seed=7)
    assert arrays.namespace == "synthetic"
    assert arrays.images.dtype == np.uint8
    assert arrays.labels.dtype == np.int64
    assert len(arrays.labels) == 500
    assert np.array_equal(arrays.images, prepare_data(tmp_path, synthetic=True, synthetic_per_class=5, seed=7).images)
    with pytest.raises(ValueError, match="namespace"):
        CifarArrays(arrays.images, arrays.labels, "train:official")
