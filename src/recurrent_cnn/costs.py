"""Unique-parameter and executed-MAC accounting for recurrent CNNs."""

from __future__ import annotations

from math import isfinite
from typing import Any

from torch import nn

from .models import ModelSpec

_SUPPORTED_DEPTHS = (1, 2, 4, 8)


def _component_counts(width: int, num_classes: int, recall: bool = False) -> dict[str, int]:
    stem = 3 * width * 3 * 3
    head = width * num_classes + num_classes
    branch = 2 * width * width * 3 * 3 + 4 * width  # two affine GroupNorm layers
    result = {"stem": stem, "head": head, "branch": branch}
    if recall:
        result["recall_projection"] = 2 * width * width
    return result


def _count_for(family: str, depth: int, width: int, num_classes: int = 100, recall: bool = False) -> int:
    counts = _component_counts(width, num_classes, recall)
    branch_copies = 3 if family == "H" else (1 if family in ("C", "E", "F") else depth)
    return counts["stem"] + counts["head"] + branch_copies * counts["branch"] + counts.get("recall_projection", 0)


def parameter_count(model: nn.Module) -> int:
    """Count unique learnable parameter objects (shared tensors are counted once)."""
    return sum(parameter.numel() for parameter in model.parameters())


def match_untied_width(depth: int, target_width: int = 32, num_classes: int = 100) -> dict[str, Any]:
    """Match an untied model to a same-depth tied model by total parameter count.

    Widths 1..32 are enumerated; equal absolute errors choose the larger width.
    ``mismatch`` is signed (untied minus tied), and ``relative_mismatch`` is
    relative to the tied target. At K=1, the families have the same count at an
    equal width, so the result explicitly identifies that degeneracy.
    """
    if depth not in (1, 2, 4, 8):
        raise ValueError("depth must be one of (1, 2, 4, 8)")
    if not 1 <= target_width <= 32:
        raise ValueError("target_width must be in 1..32")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    target = _count_for("C", depth, target_width, num_classes)
    candidates = [
        {"width": width, "parameters": _count_for("B", depth, width, num_classes)}
        for width in range(1, 33)
    ]
    best = min(candidates, key=lambda row: (abs(row["parameters"] - target), -row["width"]))
    mismatch = best["parameters"] - target
    return {
        "depth": depth,
        "target_width": target_width,
        "width": best["width"],
        "target_parameters": target,
        "matched_parameters": best["parameters"],
        "mismatch": mismatch,
        "relative_mismatch": mismatch / target,
        "k1_degenerate": depth == 1 and best["width"] == target_width,
        "candidates": candidates,
    }


def mac_breakdown(
    spec: ModelSpec,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    *,
    inference_k: int | None = None,
) -> dict[str, Any]:
    """Return JSON-compatible batch MACs and unique parameters.

    One multiply-accumulate is one MAC. Executed branch operations are counted
    on every recurrent step, including for tied weights. GroupNorm, ReLU,
    residual add/scale, pooling and memory traffic are omitted.
    """
    if len(input_shape) != 4:
        raise ValueError("input_shape must be NCHW")
    batch, channels, height, width_in = input_shape
    if min(input_shape) < 1 or channels != 3:
        raise ValueError("input_shape must have positive NCHW dimensions and 3 channels")
    k = spec.depth if inference_k is None else inference_k
    if type(k) is not int or k < 1:
        raise ValueError("inference_k must be a positive integer")
    if spec.family == "H" and k != 8:
        raise ValueError("family H supports inference_k=8 only")
    if spec.family in ("A", "B", "D") and k > spec.depth:
        raise ValueError("inference_k cannot exceed the fixed model depth")
    width = spec.width
    if spec.family == "D":
        width = match_untied_width(spec.depth, spec.width, spec.num_classes)["width"]
    recall = spec.family in ("E", "F") and spec.recall is not False
    # Conv3x3, stride2, padding1 yields ceil(H/2) x ceil(W/2).
    stem_h, stem_w = (height + 1) // 2, (width_in + 1) // 2
    stem = batch * stem_h * stem_w * width * 3 * 3 * 3
    branch_conv = batch * stem_h * stem_w * width * width * 3 * 3
    head = batch * spec.num_classes * width
    executed_branch = 2 * k * branch_conv
    recall_projection = batch * stem_h * stem_w * 2 * width * width if recall else 0
    executed_recall = k * recall_projection
    counts = _component_counts(width, spec.num_classes, recall)
    copies = 3 if spec.family == "H" else (1 if spec.family in ("C", "E", "F") else spec.depth)
    unique_parameters = counts["stem"] + counts["head"] + copies * counts["branch"] + counts.get("recall_projection", 0)
    result: dict[str, Any] = {
        "family": spec.family,
        "depth": spec.depth,
        "width": width,
        "input_shape": list(input_shape),
        "parameter_breakdown": {
            "stem": counts["stem"],
            "branch_per_unique_copy": counts["branch"],
            "unique_branch_copies": copies,
            "head": counts["head"],
            "unique_total": unique_parameters,
        },
        "mac_breakdown": {
            "stem": stem,
            "branch_conv_per_step": 2 * branch_conv,
            "executed_branch": executed_branch,
            "head": head,
            "total": stem + executed_branch + head,
        },
        "omitted": ["GroupNorm", "ReLU", "residual add/scale", "pooling", "memory traffic"],
    }
    if recall:
        result["parameter_breakdown"]["recall_projection"] = counts["recall_projection"]
        result["mac_breakdown"]["recall_projection_per_step"] = recall_projection
        result["mac_breakdown"]["executed_recall_projection"] = executed_recall
        result["mac_breakdown"]["total"] += executed_recall
    if inference_k is not None:
        result["inference_k"] = k
        result["depth"] = spec.depth
    if spec.family == "D":
        match = match_untied_width(spec.depth, spec.width, spec.num_classes)
        result["parameter_match"] = {
            key: match[key]
            for key in ("target_parameters", "matched_parameters", "mismatch", "relative_mismatch", "k1_degenerate")
        }
    return result


def match_untied_width_to_macs(
    target_spec: ModelSpec,
    *,
    target_k: int = 32,
    untied_depth: int = 8,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    max_width: int = 128,
    tolerance: float = 0.03,
) -> dict[str, Any]:
    """Find the smallest untied B width meeting a recurrent target MAC count.

    Candidates must meet or exceed the target and remain within the declared
    relative excess; unlike parameter matching this is an inference-compute
    control, not an exact-MAC claim.
    """
    if target_spec.family not in ("E", "F"):
        raise ValueError("compute matching target must be an E/F model")
    if type(untied_depth) is not int or untied_depth not in _SUPPORTED_DEPTHS:
        raise ValueError(f"untied_depth must be one of {_SUPPORTED_DEPTHS}")
    if type(max_width) is not int or max_width < 1:
        raise ValueError("max_width must be a positive integer")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    target = mac_breakdown(target_spec, input_shape, inference_k=target_k)["mac_breakdown"]["total"]
    candidates = []
    for candidate_width in range(1, max_width + 1):
        candidate = ModelSpec("B", untied_depth, candidate_width, num_classes=target_spec.num_classes)
        macs = mac_breakdown(candidate, input_shape)["mac_breakdown"]["total"]
        candidates.append({"width": candidate_width, "forward_macs": macs, "relative_excess": (macs - target) / target})
    eligible = [item for item in candidates if 0 <= item["relative_excess"] <= tolerance]
    if not eligible:
        raise ValueError("no untied width meets or exceeds target MACs within tolerance")
    selected = min(eligible, key=lambda item: item["width"])
    return {"target_forward_macs": target, "depth": untied_depth, "width": selected["width"],
            "forward_macs": selected["forward_macs"], "relative_excess": selected["relative_excess"],
            "tolerance": tolerance, "candidates": candidates}
