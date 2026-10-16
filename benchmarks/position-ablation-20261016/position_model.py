"""Mapped END/INNER position treatments on the native H parameterization."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch

from recurrent_cnn.models import ModelSpec, RecurrentCNN

BRANCH_MAPS = {
    "END": (0, 1, 1, 1, 1, 1, 1, 2),
    "INNER": (1, 0, 1, 1, 1, 1, 2, 1),
}


class PositionModel(RecurrentCNN):
    """Native-H state-dict layout with explicit position-to-branch treatment."""

    def __init__(self, spec: ModelSpec, *, seed: int | None = None, arm: str) -> None:
        if arm not in BRANCH_MAPS:
            raise ValueError("arm must be END or INNER")
        if (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) != ("H", 8, 32, 0.1, 100):
            raise ValueError("position model requires native H depth=8 width=32 alpha=0.1 classes=100")
        if any(getattr(spec, key) is not None for key in ("recall", "variable_depth", "progressive_loss")):
            raise ValueError("position model does not support additional mechanisms")
        super().__init__(spec, seed=seed)
        self.arm = arm
        self.branch_map = BRANCH_MAPS[arm]
        source = self.branches[0].state_dict()
        for branch in self.branches[1:]:
            branch.load_state_dict(deepcopy(source))

    def step(self, h, encoded=None, *, iteration=None):
        if iteration is None or type(iteration) is not int or not 0 <= iteration < 8:
            raise ValueError("position step requires iteration in 0..7")
        return h + self.spec.alpha * self.branches[self.branch_map[iteration]](h)


def build_position_model(spec: ModelSpec, *, seed: int | None, arm: str) -> PositionModel:
    return PositionModel(spec, seed=seed, arm=arm)


@contextmanager
def mapped_engine_factory(engine, arm: str):
    """Temporarily route native H factory calls through one treatment; always restore."""
    if arm not in BRANCH_MAPS:
        raise ValueError("arm must be END or INNER")
    original = engine.build_model

    def factory(spec, *, seed=None):
        if spec.family == "H":
            return build_position_model(spec, seed=seed, arm=arm)
        return original(spec, seed=seed)

    engine.build_model = factory
    try:
        yield
    finally:
        engine.build_model = original
