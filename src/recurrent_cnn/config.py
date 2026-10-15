"""Strict, canonical training configuration and stable seed derivation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import math
from typing import Any, Mapping

import yaml

from .data import canonical_json, canonical_sha256


@dataclass(frozen=True)
class TrainConfig:
    family: str
    depth: int
    width: int = 32
    alpha: float = 0.1
    fraction: float = 1.0
    paired_seed: int = 0
    data_hash: str = ""
    split_hash: str = ""
    subset_hash: str = ""
    role: str = "smoke"
    synthetic: bool = True
    total_updates: int = 2000
    batch_size: int = 128
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    cosine_horizon: int | None = None
    validation_cadence: int = 100
    device: str = "cpu"
    amp: bool = False
    num_classes: int = 100
    split_seed: int = 0
    recall: bool | None = None
    variable_depth: bool | None = None
    progressive_loss: bool | None = None
    selection_identity: str | None = None
    progressive_weight: float | None = None

    def __post_init__(self) -> None:
        if self.family not in ("A", "B", "C", "D", "E", "F", "H"):
            raise ValueError("family must be A, B, C, D, E, F, or H")
        for key in ("depth", "width", "paired_seed", "total_updates", "batch_size", "validation_cadence", "num_classes", "split_seed"):
            value = getattr(self, key)
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.depth not in (1, 2, 4, 8) or (self.family == "A" and self.depth != 1) or (self.family == "H" and self.depth != 8):
            raise ValueError("invalid family/depth combination")
        if self.width < 1 or self.num_classes < 1 or self.batch_size < 1 or self.total_updates < 1:
            raise ValueError("width, classes, batch_size and total_updates must be positive")
        if self.validation_cadence < 1 or self.split_seed < 0 or self.paired_seed < 0:
            raise ValueError("validation cadence must be positive and seeds nonnegative")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)) or not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        if self.family in ("A", "B", "C", "D", "H") and self.alpha != 0.1:
            raise ValueError("families A-D and H fix alpha at 0.1")
        if self.family in ("E", "F") and self.alpha <= 0:
            raise ValueError("families E-F require positive alpha")
        mechanisms = ("recall", "variable_depth", "progressive_loss")
        if self.family in ("A", "B", "C", "D", "H") and any(getattr(self, name) is not None for name in mechanisms):
            raise ValueError("training mechanisms are only configurable for E/F")
        if any(getattr(self, name) is not None and type(getattr(self, name)) is not bool for name in mechanisms):
            raise ValueError("training mechanism flags must be booleans when specified")
        if self.family == "E" and any(getattr(self, name) is False for name in mechanisms):
            raise ValueError("family E requires all mechanisms; use family F for ablations")
        if self.family == "F" and any(type(getattr(self, name)) is not bool for name in mechanisms):
            raise ValueError("family F requires explicit boolean mechanism flags")
        if self.selection_identity not in (None, "amended_v2"):
            raise ValueError("selection_identity must be amended_v2 when specified")
        if self.family not in ("E", "F") and self.progressive_weight is not None:
            raise ValueError("progressive_weight is only configurable for E/F")
        if self.family in ("E", "F") and (
            isinstance(self.progressive_weight, bool)
            or (self.progressive_weight is not None and (
                not isinstance(self.progressive_weight, (int, float))
                or not math.isfinite(self.progressive_weight)
                or not 0 <= self.progressive_weight <= 1
            ))
        ):
            raise ValueError("progressive_weight must be finite and in [0, 1]")
        if isinstance(self.fraction, bool) or not isinstance(self.fraction, (int, float)) or not math.isfinite(self.fraction) or not 0 < self.fraction <= 1:
            raise ValueError("fraction must be finite and in (0, 1]")
        if self.role not in ("pilot", "research", "smoke", "overfit"):
            raise ValueError("role must be pilot, research, smoke, or overfit")
        if type(self.synthetic) is not bool or type(self.amp) is not bool:
            raise ValueError("synthetic and amp must be booleans")
        if self.role == "overfit" and not self.synthetic:
            raise ValueError("overfit role requires synthetic=True")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda, or mps")
        if any(not isinstance(x, str) for x in (self.data_hash, self.split_hash, self.subset_hash)):
            raise ValueError("identity hashes must be strings")
        if self.cosine_horizon is not None and (type(self.cosine_horizon) is not int or self.cosine_horizon < 1):
            raise ValueError("cosine_horizon must be a positive integer")
        for key in ("learning_rate", "momentum", "weight_decay"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if self.learning_rate <= 0 or self.momentum < 0 or self.momentum >= 1 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive; momentum must be in [0, 1); weight_decay must be nonnegative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainConfig":
        if not isinstance(value, Mapping):
            raise ValueError("configuration must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ValueError(f"invalid configuration: {exc}") from exc

    @classmethod
    def from_yaml(cls, text: str) -> "TrainConfig":
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError("invalid YAML configuration") from exc
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cosine_horizon"] = self.cosine_horizon or self.total_updates
        if self.family in ("E", "F"):
            result.update({
                "recall": True if self.recall is None else self.recall,
                "variable_depth": True if self.variable_depth is None else self.variable_depth,
                "progressive_loss": True if self.progressive_loss is None else self.progressive_loss,
                "selection_identity": "amended_v2",
                "progressive_weight": 0.5 if self.progressive_weight is None else self.progressive_weight,
            })
        else:
            # Preserve legacy configs; amended A-D controls serialize only their selector identity.
            for key in ("recall", "variable_depth", "progressive_loss", "progressive_weight"):
                result.pop(key)
            if self.selection_identity is None:
                result.pop("selection_identity")
        return result

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def derived_seeds(self) -> dict[str, int]:
        seeds = derive_seeds(self.paired_seed, self.split_seed, self.data_hash, self.split_hash, self.subset_hash)
        if self.family in ("E", "F"):
            root = {"paired_seed": self.paired_seed, "split_seed": self.split_seed, "data_hash": self.data_hash,
                    "split_hash": self.split_hash, "subset_hash": self.subset_hash}
            seeds["recurrence_path"] = int.from_bytes(
                hashlib.sha256((canonical_json(root) + ":recurrence_path").encode()).digest()[:4], "big")
        return seeds


def derive_seeds(paired_seed: int, split_seed: int = 0, data_hash: str = "", split_hash: str = "", subset_hash: str = "") -> dict[str, int]:
    """Return actual subset sampling root plus independent stable runtime streams.

    ``data.make_subsets`` uses ``paired_seed`` as its sampling root and domains
    per-class SeedSequences internally; therefore ``subset`` records that exact
    root rather than an unused separately hashed value. Init, minibatch, and
    augmentation streams use independent canonical SHA256 derivations.
    """
    root = {"paired_seed": paired_seed, "split_seed": split_seed, "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash}
    derived = {name: int.from_bytes(hashlib.sha256((canonical_json(root) + ":" + name).encode()).digest()[:4], "big")
               for name in ("init", "minibatch", "augmentation")}
    return {"subset": paired_seed, **derived}
