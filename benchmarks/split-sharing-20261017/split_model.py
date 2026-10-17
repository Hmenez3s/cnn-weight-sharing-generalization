"""Benchmark-local five-block trunk treatments for the validation-only split study."""
from __future__ import annotations

from contextlib import contextmanager
import torch
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
    "AB2W32G1": {"family": "C", "depth": 2, "width": 32, "branch_map": None, "init_source_indices": None, "unique_blocks": 1, "parameter_count": 22_724, "macs": 9_661_568, "group_norm_groups": 1, "branch_convolutions": 2, "residual_updates": 2, "architecture": "two-convolution preactivated branch shared across two updates"},
    "A3W37G1": {"family": "C", "depth": 3, "width": 37, "branch_map": None, "init_source_indices": None, "unique_blocks": 1, "parameter_count": 17_194, "macs": 9_721_972, "group_norm_groups": 1, "branch_convolutions": 1, "residual_updates": 3, "architecture": "one-convolution preactivated branch shared across three updates"},
}
NATIVE_ARMS = ("C4W32", "B4W45", "C4W45")
GRANULARITY_ARMS = ("AB2W32G1", "A3W37G1")
_SIGNATURES = {"AB2W32G1": (0x434752, 1, 1, 2, 1, 2, 32), "A3W37G1": (0x434752, 1, 2, 1, 1, 3, 37)}
EXPECTED_PARAMETERS = 96_964


class _TwoConvGroupOneBranch(nn.Module):
    def __init__(self, width, conv1, conv2):
        super().__init__()
        self.norm1, self.relu1, self.conv1 = nn.GroupNorm(1, width), nn.ReLU(inplace=False), conv1
        self.norm2, self.relu2, self.conv2 = nn.GroupNorm(1, width), nn.ReLU(inplace=False), conv2

    def forward(self, x):
        return self.conv2(self.relu2(self.norm2(self.conv1(self.relu1(self.norm1(x))))))


class _OneConvGroupOneBranch(nn.Module):
    def __init__(self, width, conv):
        super().__init__()
        self.norm1, self.relu1, self.conv1 = nn.GroupNorm(1, width), nn.ReLU(inplace=False), conv

    def forward(self, x):
        return self.conv1(self.relu1(self.norm1(x)))


def arm_cost_breakdown(arm, input_shape=(1, 3, 32, 32), *, inference_k=None):
    """Authoritative analytic cost for the two opt-in granularity arms."""
    if arm not in GRANULARITY_ARMS:
        raise ValueError(f"unknown granularity arm: {arm}")
    metadata = ARM_METADATA[arm]
    if len(input_shape) != 4:
        raise ValueError("input_shape must be NCHW")
    batch, channels, height, width_in = input_shape
    if min(input_shape) < 1 or channels != 3:
        raise ValueError("input_shape must have positive NCHW dimensions and 3 channels")
    k = metadata["depth"] if inference_k is None else inference_k
    if type(k) is not int or k < 1:
        raise ValueError("inference_k must be a positive integer")
    width = metadata["width"]
    out_h, out_w = (height + 1) // 2, (width_in + 1) // 2
    stem = batch * out_h * out_w * 3 * width * 9
    conv_per_step = metadata["branch_convolutions"] * batch * out_h * out_w * width * width * 9
    executed = k * conv_per_step
    head = batch * width * 100
    stem_params = 27 * width
    branch_params = metadata["branch_convolutions"] * (9 * width * width + 2 * width)
    head_params = width * 100 + 100
    return {
        "family": "C", "depth": metadata["depth"], "width": width,
        "input_shape": list(input_shape), "inference_k": k,
        "parameter_breakdown": {"stem": stem_params, "branch_per_unique_copy": branch_params,
                                "unique_branch_copies": 1, "head": head_params,
                                "unique_total": stem_params + branch_params + head_params},
        "mac_breakdown": {"stem": stem, "branch_conv_per_step": conv_per_step,
                          "executed_branch": executed, "head": head, "total": stem + executed + head},
        "omitted": ["GroupNorm", "ReLU", "residual add/scale", "pooling", "memory traffic"],
    }


def build_split_model(spec, *, seed=None, arm):
    if arm not in ARM_METADATA:
        raise ValueError(f"unknown arm: {arm}")
    metadata = ARM_METADATA[arm]
    if arm in GRANULARITY_ARMS:
        expected = ("C", metadata["depth"], metadata["width"], 0.1, 100)
        if (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) != expected:
            raise ValueError(f"{arm} requires family C depth={metadata['depth']} width={metadata['width']} alpha=0.1 classes=100")
        if any(getattr(spec, key) is not None for key in ("recall", "variable_depth", "progressive_loss")):
            raise ValueError("granularity treatment does not support additional mechanisms")
        from recurrent_cnn.models import RecurrentCNN

        class _GranularityModel(RecurrentCNN):
            def __init__(self):
                super().__init__(spec, seed=seed)
                native = self.branches[0]
                branch = (_TwoConvGroupOneBranch(spec.width, native.conv1, native.conv2)
                          if metadata["branch_convolutions"] == 2
                          else _OneConvGroupOneBranch(spec.width, native.conv1))
                self.branches = nn.ModuleList([branch])
                self.register_buffer("_architecture_signature", torch.tensor(_SIGNATURES[arm], dtype=torch.int64))
                self.arm = arm

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
                key = prefix + "_architecture_signature"
                signature = state_dict.get(key)
                if signature is not None and not torch.equal(signature.cpu(), self._architecture_signature.cpu()):
                    error_msgs.append(f"{arm} checkpoint architecture signature mismatch")
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                              missing_keys, unexpected_keys, error_msgs)

        model = _GranularityModel()
        actual = sum(parameter.numel() for parameter in model.parameters())
        expected_parameters = metadata["parameter_count"]
        if actual != expected_parameters:
            raise AssertionError(f"{arm} parameter count mismatch: {actual} != {expected_parameters}")
        return model
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
        experimental_spec = arm in GRANULARITY_ARMS and (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) == (
            "C", metadata["depth"], metadata["width"], 0.1, 100)
        if experimental_spec:
            return build_split_model(spec, seed=seed, arm=arm)
        if metadata["branch_map"] is not None and mapped_spec:
            model = build_split_model(spec, seed=seed, arm=arm)
            actual = sum(p.numel() for p in model.parameters())
            if actual != metadata["parameter_count"]:
                raise AssertionError(f"{arm} parameter count mismatch: {actual}")
            return model
        return original_factory(spec, seed=seed)

    def split_cost(spec, input_shape=(1, 3, 32, 32), **kwargs):
        if arm in GRANULARITY_ARMS and (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) == (
                "C", metadata["depth"], metadata["width"], 0.1, 100):
            result = arm_cost_breakdown(arm, input_shape, **kwargs)
            actual = result["parameter_breakdown"]["unique_total"]
            if actual != metadata["parameter_count"]:
                raise AssertionError(f"{arm} cost parameter count mismatch: {actual}")
            return result
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
