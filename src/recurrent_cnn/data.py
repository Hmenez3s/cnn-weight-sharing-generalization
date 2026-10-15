"""Deterministic CIFAR-100 identities and explicitly acquired data arrays.

This module deliberately has no eager torchvision import: importing the data
identity utilities never accesses a dataset or triggers acquisition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


NUM_CLASSES = 100
OFFICIAL_TRAIN_PER_CLASS = 500
OFFICIAL_TEST_PER_CLASS = 100
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
IDENTITY_ALGORITHM = "cifar100-stratified-v1"
CIFAR100C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)
CIFAR100C_PER_SEVERITY = 10_000


@dataclass(frozen=True)
class CifarArrays:
    """Raw CIFAR images and labels; images are uint8 NHWC, never normalized."""

    images: np.ndarray
    labels: np.ndarray
    namespace: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        images = np.asarray(self.images)
        labels = np.asarray(self.labels)
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError("images must be uint8 NHWC with shape (N, 32, 32, 3)")
        if labels.dtype != np.int64 or labels.ndim != 1 or len(labels) != len(images):
            raise ValueError("labels must be int64 with shape (N,) matching images")
        if not isinstance(self.namespace, str) or not self.namespace.strip() or ":" in self.namespace:
            raise ValueError("namespace must be a nonempty name without ':'")
        if np.any(labels < 0) or np.any(labels >= NUM_CLASSES):
            raise ValueError("labels must be CIFAR-100 class IDs in [0, 99]")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping when supplied")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(namespace_id(self.namespace, i) for i in range(len(self.labels)))


def namespace_id(namespace: str, index: int) -> str:
    """Return a stable namespaced identity (e.g. ``train:7``)."""
    if not isinstance(namespace, str) or not namespace.strip() or ":" in namespace:
        raise ValueError("namespace must be a nonempty name without ':'")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    return f"{namespace}:{int(index)}"


def make_split(labels: Iterable[int], split_seed: int, *, synthetic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Make a fixed, class-stratified train/validation split.

    Official mode requires exactly 500 examples of each of 100 classes and
    allocates 450/50 per class. Synthetic mode retains stratification but
    requires at least two examples per class and is explicitly marked by the
    caller's manifest.
    """
    y = _labels(labels)
    _seed(split_seed, "split_seed")
    classes, counts = np.unique(y, return_counts=True)
    if not synthetic:
        if not np.array_equal(classes, np.arange(NUM_CLASSES)) or not np.all(counts == OFFICIAL_TRAIN_PER_CLASS):
            raise ValueError("official split requires exactly 500 labels for each CIFAR-100 class")
        validation_per_class = 50
    else:
        if len(classes) < 2 or np.any(counts < 2):
            raise ValueError("synthetic split requires at least two examples in each of at least two classes")
        validation_per_class = {int(c): max(1, int(n) // 10) for c, n in zip(classes, counts)}

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for class_id in classes:
        indices = np.flatnonzero(y == class_id)
        rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), int(class_id), 0x51A17]))
        order = rng.permutation(indices)
        n_val = 50 if not synthetic else validation_per_class[int(class_id)]
        val_parts.append(order[:n_val])
        train_parts.append(order[n_val:])
    train = np.sort(np.concatenate(train_parts)).astype(np.int64, copy=False)
    validation = np.sort(np.concatenate(val_parts)).astype(np.int64, copy=False)
    return train, validation


def make_subsets(
    labels: Iterable[int], train_indices: Iterable[int], paired_seed: int, *, synthetic: bool = False
) -> dict[float, np.ndarray]:
    """Create nested per-class subsets from the fixed training partition."""
    y = _labels(labels)
    train = _indices(train_indices, len(y), "train_indices")
    _seed(paired_seed, "paired_seed")
    classes, counts = np.unique(y[train], return_counts=True)
    if not synthetic:
        if not np.array_equal(classes, np.arange(NUM_CLASSES)) or not np.all(counts == 450):
            raise ValueError("official subsets require 450 training examples for each CIFAR-100 class")
        sizes = {0.10: 45, 0.25: 0, 0.50: 225, 1.00: 450}
    else:
        if len(classes) < 2 or np.any(counts < 2):
            raise ValueError("synthetic subsets require at least two training examples in each of two classes")
        sizes = {}

    # Seed one independently shuffled order per class; every smaller fraction
    # takes a prefix of that order, making nesting exact by construction.
    orders: dict[int, np.ndarray] = {}
    for class_id in classes:
        candidates = train[y[train] == class_id]
        rng = np.random.default_rng(np.random.SeedSequence([int(paired_seed), int(class_id), 0x5AB5E7]))
        orders[int(class_id)] = rng.permutation(candidates)

    if not synthetic:
        # Exactly 50 seeded classes receive the extra sample at 25%.
        class_order_rng = np.random.default_rng(np.random.SeedSequence([int(paired_seed), 0x25C1A55]))
        extra_classes = set(class_order_rng.permutation(np.arange(NUM_CLASSES))[:50].tolist())
        sizes[0.25] = 112
    else:
        extra_classes = set()

    result: dict[float, np.ndarray] = {}
    for fraction in FRACTIONS:
        selected: list[np.ndarray] = []
        for class_id, count in zip(classes, counts):
            if synthetic:
                n = max(1, int(np.floor(int(count) * fraction)))
            elif fraction == 0.25:
                n = 113 if int(class_id) in extra_classes else 112
            else:
                n = sizes[fraction]
            selected.append(orders[int(class_id)][:n])
        result[fraction] = np.sort(np.concatenate(selected)).astype(np.int64, copy=False)
    return result


def build_manifest(
    *, split_seed: int, paired_seed: int, train_indices: Iterable[int],
    validation_indices: Iterable[int], subsets: dict[float, Iterable[int]],
    synthetic: bool = False, train_namespace: str = "train",
) -> dict[str, Any]:
    """Build a canonical, self-hashing identity manifest (independent of paths)."""
    _seed(split_seed, "split_seed")
    _seed(paired_seed, "paired_seed")
    train = _indices(train_indices, None, "train_indices")
    validation = _indices(validation_indices, None, "validation_indices")
    if set(train.tolist()) & set(validation.tolist()):
        raise ValueError("train and validation indices must be disjoint")
    if not isinstance(train_namespace, str) or not train_namespace.strip() or ":" in train_namespace:
        raise ValueError("train_namespace must be a nonempty name without ':'")
    if train_namespace != ("synthetic" if synthetic else "train"):
        raise ValueError("training namespace must match the explicit synthetic mode")
    if set(subsets) != set(FRACTIONS):
        raise ValueError("subsets must contain exactly fractions 0.1, 0.25, 0.5, and 1.0")
    canonical_subsets: dict[str, list[int]] = {}
    previous: set[int] | None = None
    for fraction in FRACTIONS:
        indices = _indices(subsets[fraction], None, f"subset {fraction}")
        current = set(indices.tolist())
        if not current <= set(train.tolist()):
            raise ValueError(f"subset {fraction} is not contained in training indices")
        if previous is not None and not previous <= current:
            raise ValueError("subsets must be nested in increasing fraction order")
        previous = current
        canonical_subsets[_fraction_key(fraction)] = indices.tolist()
    if not synthetic and (len(train) != 45_000 or len(validation) != 5_000 or
                          [len(canonical_subsets[_fraction_key(f)]) for f in FRACTIONS] != [4_500, 11_250, 22_500, 45_000]):
        raise ValueError("official manifest must encode exact CIFAR-100 split and subset counts")
    body: dict[str, Any] = {
        "schema": "recurrent-cnn-data-manifest-v1",
        "algorithm": IDENTITY_ALGORITHM,
        "synthetic": bool(synthetic),
        "namespaces": {"train": train_namespace, "validation": train_namespace},
        "split_seed": int(split_seed),
        "paired_seed": int(paired_seed),
        "counts": {
            "train": len(train), "validation": len(validation),
            "subsets": {key: len(value) for key, value in canonical_subsets.items()},
        },
        "indices": {"train": train.tolist(), "validation": validation.tolist(), "subsets": canonical_subsets},
    }
    body["sha256"] = canonical_sha256(body)
    return body


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically using UTF-8-compatible canonical form."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Validate schema, index/count consistency, nesting, and content hash."""
    if not isinstance(manifest, dict) or manifest.get("schema") != "recurrent-cnn-data-manifest-v1":
        raise ValueError("unsupported or malformed data manifest")
    claimed_hash = manifest.get("sha256")
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if not isinstance(claimed_hash, str) or canonical_sha256(body) != claimed_hash:
        raise ValueError("manifest SHA256 mismatch")
    try:
        train = _indices(body["indices"]["train"], None, "manifest train")
        validation = _indices(body["indices"]["validation"], None, "manifest validation")
        subsets_raw = body["indices"]["subsets"]
        if set(subsets_raw) != {_fraction_key(f) for f in FRACTIONS}:
            raise ValueError("manifest subset fractions are invalid")
        if set(train.tolist()) & set(validation.tolist()):
            raise ValueError("manifest train/validation overlap")
        subset_sets: list[set[int]] = []
        for f in FRACTIONS:
            indices = _indices(subsets_raw[_fraction_key(f)], None, f"manifest subset {f}")
            current = set(indices.tolist())
            if not current <= set(train.tolist()) or (subset_sets and not subset_sets[-1] <= current):
                raise ValueError("manifest subset is non-nested or outside training partition")
            subset_sets.append(current)
        expected = {"train": len(train), "validation": len(validation), "subsets": {
            _fraction_key(f): len(subset_sets[i]) for i, f in enumerate(FRACTIONS)}}
        if body.get("counts") != expected:
            raise ValueError("manifest counts do not match indices")
        if not body.get("synthetic") and (expected["train"] != 45_000 or expected["validation"] != 5_000 or
                list(expected["subsets"].values()) != [4_500, 11_250, 22_500, 45_000]):
            raise ValueError("official manifest must encode exact CIFAR-100 split and subset counts")
        if body.get("algorithm") != IDENTITY_ALGORITHM or not isinstance(body.get("synthetic"), bool):
            raise ValueError("manifest algorithm or synthetic marker is invalid")
        _seed(body["split_seed"], "split_seed")
        _seed(body["paired_seed"], "paired_seed")
        namespaces = body["namespaces"]
        expected_train_namespace = "synthetic" if body["synthetic"] else "train"
        if not isinstance(namespaces, dict) or set(namespaces) != {"train", "validation"} or (
            namespaces.get("train") != expected_train_namespace
            or namespaces.get("validation") != expected_train_namespace
        ):
            raise ValueError("manifest namespaces do not match the explicit mode")
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed data manifest") from exc
    return manifest


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Persist a validated immutable manifest; existing files are never replaced."""
    validated = validate_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(validated) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return target


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read data manifest") from exc
    return validate_manifest(value)


def prepare_data(root: str | Path, download: bool = False, *, synthetic: bool = False,
                 synthetic_per_class: int = 20, seed: int = 0) -> CifarArrays:
    """Load only CIFAR-100 training data, or explicitly create marked synthetic data.

    Official acquisition is possible only with ``download=True``. Synthetic
    mode never imports torchvision or touches ``root``.
    """
    if not isinstance(download, bool) or not isinstance(synthetic, bool):
        raise ValueError("download and synthetic must be explicit booleans")
    if synthetic:
        if synthetic_per_class < 2:
            raise ValueError("synthetic_per_class must be at least 2")
        _seed(seed, "seed")
        rng = np.random.default_rng(seed)
        images = rng.integers(0, 256, size=(NUM_CLASSES * synthetic_per_class, 32, 32, 3), dtype=np.uint8)
        labels = np.repeat(np.arange(NUM_CLASSES, dtype=np.int64), synthetic_per_class)
        return CifarArrays(images=images, labels=labels, namespace="synthetic")
    try:
        from torchvision.datasets import CIFAR100
    except ImportError as exc:
        raise RuntimeError("torchvision is required only when loading official CIFAR-100") from exc
    # torchvision otherwise opens the test file for its integrity checksum even
    # with train=True. Keep that partition closed until protocol authorization.
    class TrainingCIFAR100(CIFAR100):
        test_list = []

    dataset = TrainingCIFAR100(root=str(root), train=True, download=download)
    images = np.asarray(dataset.data, dtype=np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    arrays = CifarArrays(images=images, labels=labels, namespace="train")
    if len(labels) != NUM_CLASSES * OFFICIAL_TRAIN_PER_CLASS:
        raise ValueError("official CIFAR-100 training partition must contain exactly 50,000 examples")
    return arrays


def make_validation_corruption(
    arrays: CifarArrays, validation_indices: Iterable[int], corruption: str, severity: int,
    seed: int, *, split_seed: int = 1729,
) -> CifarArrays:
    """Create deterministic held-out validation corruption (not CIFAR-100-C).

    Indices must exactly match the canonical validation partition. The source
    is immutable; randomness is local to this call and never affects training.
    """
    if not isinstance(arrays, CifarArrays) or arrays.namespace not in ("train", "synthetic"):
        raise ValueError("validation corruption requires canonical train or synthetic source arrays")
    if corruption not in ("gaussian_blur", "saturate", "spatter"):
        raise ValueError("unknown validation corruption")
    if type(severity) is not int or severity not in range(1, 6):
        raise ValueError("severity must be an integer in [1, 5]")
    _seed(seed, "seed")
    _seed(split_seed, "split_seed")
    indices = _indices(validation_indices, len(arrays.labels), "validation_indices")
    expected = make_split(arrays.labels, split_seed, synthetic=arrays.namespace == "synthetic")[1]
    if not np.array_equal(indices, expected):
        raise ValueError("indices must be the canonical validation partition, not a training population")
    source = arrays.images[indices]
    source_labels = arrays.labels[indices]
    origin_hash = hashlib.sha256()
    origin_hash.update(np.ascontiguousarray(source).view(np.uint8))
    origin_hash.update(np.ascontiguousarray(source_labels).view(np.uint8))
    origin_identity = origin_hash.hexdigest()
    rng = np.random.default_rng(np.random.SeedSequence([seed, severity, 0xC0A]))
    images = _validation_transform(source, corruption, severity, rng)
    return CifarArrays(images, source_labels.astype(np.int64, copy=True), "validation_corruption", {
        "source": "heldout_validation", "origin_namespace": arrays.namespace,
        "origin_identity": origin_identity,
        "validation_indices_sha256": canonical_sha256(indices.tolist()),
        "transform": corruption, "severity": severity, "seed": seed,
        "evaluation_namespace": f"validation-corruption-{corruption}-s{severity}",
        "identity": canonical_sha256({"origin": origin_identity, "namespace": arrays.namespace,
                                       "indices": indices.tolist(), "transform": corruption,
                                       "severity": severity, "seed": seed}),
    })


def _validation_transform(images: np.ndarray, corruption: str, severity: int,
                          rng: np.random.Generator) -> np.ndarray:
    result = np.empty_like(images)
    for start in range(0, len(images), 128):
        stop = min(start + 128, len(images))
        result[start:stop] = _validation_transform_batch(images[start:stop], corruption, severity, rng)
    return result


def _validation_transform_batch(images: np.ndarray, corruption: str, severity: int,
                                rng: np.random.Generator) -> np.ndarray:
    source = images.astype(np.float32)
    if corruption == "gaussian_blur":
        sigma = 0.35 + 0.30 * severity
        radius = min(3, severity)
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(axis * axis) / (2.0 * sigma * sigma)); kernel /= kernel.sum()
        padded = np.pad(source, ((0, 0), (radius, radius), (radius, radius), (0, 0)), mode="reflect")
        horizontal = sum(float(weight) * padded[:, radius:radius + 32, radius + offset:radius + offset + 32]
                         for offset, weight in zip(range(-radius, radius + 1), kernel))
        padded_horizontal = np.pad(horizontal, ((0, 0), (radius, radius), (0, 0), (0, 0)), mode="reflect")
        blurred = sum(float(weight) * padded_horizontal[:, radius + offset:radius + offset + 32]
                      for offset, weight in zip(range(-radius, radius + 1), kernel))
        result = blurred
    elif corruption == "saturate":
        strength = 0.25 * severity
        gray = np.sum(source * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=-1, keepdims=True)
        result = gray + (source - gray) * (1.0 + strength)
    else:
        amount = 0.025 * severity
        droplets = rng.random(source.shape[:3]) < amount
        color = rng.choice(np.array([0.0, 255.0], dtype=np.float32), size=source.shape[:3])
        result = source * (1.0 - droplets[..., None] * 0.55) + color[..., None] * droplets[..., None] * 0.55
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _open_cifar100_c(root: str | Path, corruption: str, severity: int) -> CifarArrays:
    """Lazily open one strict local CIFAR-100-C condition, read-only."""
    if corruption not in CIFAR100C_CORRUPTIONS:
        raise ValueError("unknown CIFAR-100-C corruption")
    if type(severity) is not int or severity not in range(1, 6):
        raise ValueError("severity must be an integer in [1, 5]")
    base = Path(root)
    try:
        images = np.load(base / f"{corruption}.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(base / "labels.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read strict CIFAR-100-C NPY files") from exc
    expected_count = 5 * CIFAR100C_PER_SEVERITY
    if not isinstance(images, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError("CIFAR-100-C files must be NPY arrays")
    if images.dtype != np.uint8 or images.shape != (expected_count, 32, 32, 3):
        raise ValueError("CIFAR-100-C images must be uint8 (50000, 32, 32, 3)")
    if (labels.ndim != 1 or labels.shape != (expected_count,) or
            not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_)):
        raise ValueError("CIFAR-100-C labels must be an integer vector of length 50000")
    if np.any(labels < 0) or np.any(labels >= NUM_CLASSES):
        raise ValueError("CIFAR-100-C labels must be in [0, 99]")
    label_blocks = labels.reshape(5, CIFAR100C_PER_SEVERITY)
    if not all(np.array_equal(label_blocks[0], label_blocks[i]) for i in range(1, 5)):
        raise ValueError("CIFAR-100-C severity label blocks must repeat exactly")
    counts = np.bincount(label_blocks[0].astype(np.int64, copy=False), minlength=NUM_CLASSES)
    if not np.all(counts == OFFICIAL_TEST_PER_CLASS):
        raise ValueError("CIFAR-100-C requires 100 examples per class in each severity block")
    start = (severity - 1) * CIFAR100C_PER_SEVERITY
    image_view = images[start:start + CIFAR100C_PER_SEVERITY]
    label_view = labels[start:start + CIFAR100C_PER_SEVERITY].astype(np.int64, copy=True)
    image_view.setflags(write=False)
    label_view.setflags(write=False)
    return CifarArrays(image_view, label_view, "cifar100c", {
        "source": "cifar100c", "origin_namespace": "official_test_corruption",
        "corruption": corruption, "severity": severity,
        "evaluation_namespace": f"cifar100c-{corruption}-s{severity}",
        "image_offset": start, "count": CIFAR100C_PER_SEVERITY,
    })


def _open_official_test(root: str | Path, download: bool = False) -> CifarArrays:
    """Open CIFAR-100 test only when an authorized protocol explicitly calls it."""
    if not isinstance(download, bool):
        raise ValueError("download must be a boolean")
    try:
        from torchvision.datasets import CIFAR100
    except ImportError as exc:
        raise RuntimeError("torchvision is required only when loading official CIFAR-100") from exc
    dataset = CIFAR100(root=str(root), train=False, download=download)
    return CifarArrays(
        images=np.asarray(dataset.data, dtype=np.uint8),
        labels=np.asarray(dataset.targets, dtype=np.int64),
        namespace="test",
    )


def _labels(labels: Iterable[int]) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError("labels must be a one-dimensional integer sequence")
    result = array.astype(np.int64, copy=False)
    if np.any(result < 0) or np.any(result >= NUM_CLASSES):
        raise ValueError("labels must be class IDs in [0, 99]")
    return result


def _indices(values: Iterable[int], upper: int | None, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    result = array.astype(np.int64, copy=False)
    if len(np.unique(result)) != len(result) or np.any(result < 0) or (upper is not None and np.any(result >= upper)):
        raise ValueError(f"{name} must contain unique in-range nonnegative indices")
    return np.sort(result)


def _seed(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0 or value >= 2**32:
        raise ValueError(f"{name} must be an integer in [0, 2**32)")


def _fraction_key(fraction: float) -> str:
    return f"{fraction:.2f}"
