"""Benchmark-local five-block trunk treatments for the validation-only split study."""
from __future__ import annotations

from contextlib import contextmanager

BRANCH_MAPS = {
    "C4B4": (0, 0, 0, 0, 1, 2, 3, 4),
    "B4C4": (0, 1, 2, 3, 4, 4, 4, 4),
}
EXPECTED_PARAMETERS = 96_964


def build_split_model(spec, *, seed=None, arm):
    if arm not in BRANCH_MAPS:
        raise ValueError("arm must be C4B4 or B4C4")
    if (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) != ("B", 8, 32, 0.1, 100):
        raise ValueError("split treatment requires native B depth=8 width=32 alpha=0.1 classes=100")
    if any(getattr(spec, key) is not None for key in ("recall", "variable_depth", "progressive_loss")):
        raise ValueError("split treatment does not support additional mechanisms")
    from recurrent_cnn.models import RecurrentCNN

    class _SplitModel(RecurrentCNN):
        def __init__(self):
            super().__init__(spec, seed=seed)
            del self.branches[5:]
            self.arm = arm
            self.branch_map = BRANCH_MAPS[arm]

        def step(self, h, encoded=None, *, iteration=None):
            if iteration is None or type(iteration) is not int or not 0 <= iteration < 8:
                raise ValueError("split step requires iteration in 0..7")
            return h + self.spec.alpha * self.branches[self.branch_map[iteration]](h)

    return _SplitModel()


@contextmanager
def mapped_engine_factory(engine, arm: str):
    """Route native B factory through treatment and correct only analytic unique-block count."""
    if arm not in BRANCH_MAPS:
        raise ValueError("arm must be C4B4 or B4C4")
    original_factory, original_cost = engine.build_model, engine.mac_breakdown

    def factory(spec, *, seed=None):
        if spec.family == "B":
            model = build_split_model(spec, seed=seed, arm=arm)
            if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETERS:
                raise AssertionError("split model parameter count mismatch")
            return model
        return original_factory(spec, seed=seed)

    def split_cost(spec, input_shape=(1, 3, 32, 32), **kwargs):
        result = original_cost(spec, input_shape, **kwargs)
        if spec.family == "B":
            # mac_breakdown returns batch MACs; the split trunk executes the same eight
            # block applications as native B8, i.e. 37,973,120 MACs per 32x32 image.
            if tuple(input_shape[1:]) == (3, 32, 32) and result["mac_breakdown"]["total"] != 37_973_120 * input_shape[0]:
                raise AssertionError("split model MAC count differs from native B8")
            result["parameter_breakdown"]["unique_branch_copies"] = 5
            result["parameter_breakdown"]["unique_total"] = EXPECTED_PARAMETERS
        return result

    engine.build_model, engine.mac_breakdown = factory, split_cost
    try:
        yield
    finally:
        engine.build_model, engine.mac_breakdown = original_factory, original_cost
