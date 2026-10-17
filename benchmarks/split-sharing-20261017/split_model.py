"""Benchmark-local five-block trunk treatments for the validation-only split study."""
from __future__ import annotations

from contextlib import contextmanager
from torch import nn

BRANCH_MAPS = {
    "C4B4": (0, 0, 0, 0, 1, 2, 3, 4),
    "B4C4": (0, 1, 2, 3, 4, 4, 4, 4),
    "C7B1": (0, 0, 0, 0, 0, 0, 0, 1),
    "C4W45": None,
    "C4W32": None,
    "B4W45": None,
}
ARM_METADATA = {
    "C4B4": {"family": "B", "depth": 8, "width": 32, "branch_map": BRANCH_MAPS["C4B4"], "init_source_indices": [0, 1, 2, 3, 4], "unique_blocks": 5, "parameter_count": 96_964, "macs": 37_973_120},
    "B4C4": {"family": "B", "depth": 8, "width": 32, "branch_map": BRANCH_MAPS["B4C4"], "init_source_indices": [0, 1, 2, 3, 4], "unique_blocks": 5, "parameter_count": 96_964, "macs": 37_973_120},
    "C7B1": {"family": "B", "depth": 8, "width": 32, "branch_map": BRANCH_MAPS["C7B1"], "init_source_indices": [1, 2], "unique_blocks": 2, "parameter_count": 41_284, "macs": 37_973_120, "init_convention": "H8-matched; native B8 branches 1 and 2 retained (H8 S,Z)"},
    "C4W45": {"family": "C", "depth": 4, "width": 45, "branch_map": None, "init_source_indices": None, "unique_blocks": 1, "parameter_count": 42_445, "macs": 37_640_340, "target_macs": 37_973_120, "mac_relative_mismatch": -0.00876356749, "group_norm_groups": 1},
    "C4W32": {"family": "C", "depth": 4, "width": 32, "branch_map": None, "init_source_indices": None, "unique_blocks": 1, "parameter_count": 22_724, "macs": 19_098_752, "group_norm_groups": 8},
    "B4W45": {"family": "B", "depth": 4, "width": 45, "branch_map": None, "init_source_indices": None, "unique_blocks": 4, "parameter_count": 152_335, "macs": 37_640_340, "group_norm_groups": 1},
}
NATIVE_ARMS = ("C4W32", "B4W45", "C4W45")
EXPECTED_PARAMETERS = 96_964


def build_split_model(spec, *, seed=None, arm):
    if arm not in ARM_METADATA:
        raise ValueError(f"unknown arm: {arm}")
    metadata = ARM_METADATA[arm]
    if arm in NATIVE_ARMS:
        from recurrent_cnn.models import build_model
        expected = (metadata["family"], metadata["depth"], metadata["width"], 0.1, 100)
        if (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) != expected:
            raise ValueError(f"{arm} requires native {expected[0]} depth={expected[1]} width={expected[2]} alpha=0.1 classes=100")
        return build_model(spec, seed=seed)
    if (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) != ("B", 8, 32, 0.1, 100):
        raise ValueError(f"{arm} requires native B depth=8 width=32 alpha=0.1 classes=100")
    if any(getattr(spec, key) is not None for key in ("recall", "variable_depth", "progressive_loss")):
        raise ValueError("split treatment does not support additional mechanisms")
    from recurrent_cnn.models import RecurrentCNN

    class _SplitModel(RecurrentCNN):
        def __init__(self):
            super().__init__(spec, seed=seed)
            indices = metadata["init_source_indices"]
            self.branches = nn.ModuleList([self.branches[i] for i in indices])
            self.arm = arm
            self.branch_map = metadata["branch_map"]

        def step(self, h, encoded=None, *, iteration=None):
            if iteration is None or type(iteration) is not int or not 0 <= iteration < 8:
                raise ValueError("split step requires iteration in 0..7")
            return h + self.spec.alpha * self.branches[self.branch_map[iteration]](h)

    return _SplitModel()


@contextmanager
def mapped_engine_factory(engine, arm: str):
    """Route native B factory through treatment and correct only analytic unique-block count."""
    if arm not in ARM_METADATA:
        raise ValueError(f"unknown arm: {arm}")
    metadata = ARM_METADATA[arm]
    original_factory, original_cost = engine.build_model, engine.mac_breakdown

    def factory(spec, *, seed=None):
        mapped_spec = (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) == ("B", 8, 32, 0.1, 100)
        if metadata["branch_map"] is not None and mapped_spec:
            model = build_split_model(spec, seed=seed, arm=arm)
            actual = sum(p.numel() for p in model.parameters())
            if actual != metadata["parameter_count"]:
                raise AssertionError(f"{arm} parameter count mismatch: {actual}")
            return model
        return original_factory(spec, seed=seed)

    def split_cost(spec, input_shape=(1, 3, 32, 32), **kwargs):
        result = original_cost(spec, input_shape, **kwargs)
        mapped_spec = (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) == ("B", 8, 32, 0.1, 100)
        if metadata["branch_map"] is not None and mapped_spec:
            result["parameter_breakdown"]["unique_branch_copies"] = metadata["unique_blocks"]
            result["parameter_breakdown"]["unique_total"] = metadata["parameter_count"]
            if tuple(input_shape[1:]) == (3, 32, 32) and kwargs.get("inference_k", 8) == 8:
                expected_macs = metadata["macs"] * input_shape[0]
                if result["mac_breakdown"]["total"] != expected_macs:
                    raise AssertionError(f"{arm} counted MAC mismatch: {result['mac_breakdown']['total']} != {expected_macs}")
        return result

    engine.build_model, engine.mac_breakdown = factory, split_cost
    try:
        yield
    finally:
        engine.build_model, engine.mac_breakdown = original_factory, original_cost
