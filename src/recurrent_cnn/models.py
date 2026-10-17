"""Small recurrent CNN models used by the CIFAR-100 study."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd, isfinite
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Family = Literal["A", "B", "C", "D", "E", "F", "H"]
_SUPPORTED_DEPTHS = (1, 2, 3, 4, 8)
_RECURRENT_FAMILIES = ("C", "E", "F")


@dataclass(frozen=True)
class ModelSpec:
    family: Family
    depth: int
    width: int = 32
    alpha: float = 0.1
    num_classes: int = 100
    recall: bool | None = None
    variable_depth: bool | None = None
    progressive_loss: bool | None = None

    def __post_init__(self) -> None:
        if self.family not in ("A", "B", "C", "D", "E", "F", "H"):
            raise ValueError(f"unsupported model family: {self.family!r}")
        for name, value in (("depth", self.depth), ("width", self.width), ("num_classes", self.num_classes)):
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer (booleans are not accepted)")
        if self.depth not in _SUPPORTED_DEPTHS or (self.depth == 3 and self.family != "C"):
            supported = _SUPPORTED_DEPTHS if self.family == "C" else tuple(depth for depth in _SUPPORTED_DEPTHS if depth != 3)
            raise ValueError(f"depth must be one of {supported}")
        if self.family == "H" and self.depth != 8:
            raise ValueError("family H requires depth=8")
        if self.family == "A" and self.depth != 1:
            raise ValueError("family A is the single-step baseline and requires depth=1")
        if self.width < 1 or self.num_classes < 1:
            raise ValueError("width and num_classes must be positive")
        if not isinstance(self.alpha, (int, float)) or isinstance(self.alpha, bool):
            raise ValueError("alpha must be a finite positive number")
        if self.family in ("A", "B", "C", "D", "H") and self.alpha != 0.1:
            raise ValueError("families A-D and H fix residual alpha at 0.1")
        if self.family in ("E", "F") and (not isfinite(float(self.alpha)) or self.alpha <= 0):
            raise ValueError("families E-F require finite positive alpha")
        for name in ("recall", "variable_depth", "progressive_loss"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be a boolean when specified")
        if self.family not in ("E", "F") and any(
            getattr(self, name) is not None for name in ("recall", "variable_depth", "progressive_loss")
        ):
            raise ValueError("training mechanism flags are only supported for families E and F")
        if self.family == "E" and any(
            getattr(self, name) is False for name in ("recall", "variable_depth", "progressive_loss")
        ):
            raise ValueError("family E is the full-mechanism model; use family F for ablations")


class RecurrentCNN(nn.Module):
    """CNN with a constant-resolution recurrent residual branch."""

    def __init__(self, spec: ModelSpec, *, seed: int | None = None) -> None:
        super().__init__()
        if spec.family == "D":
            from .costs import match_untied_width

            matched = match_untied_width(spec.depth, spec.width, spec.num_classes)
            spec = replace(spec, width=matched["width"])
        self.spec = spec
        self.resolved_width = spec.width
        self.recall_enabled = spec.family in ("E", "F") and spec.recall is not False
        # Module defaults consume the global RNG; isolate those draws so model
        # family/depth cannot perturb independent minibatch or augmentation RNG.
        with torch.random.fork_rng(devices=[]):
            self.stem = nn.Conv2d(3, spec.width, kernel_size=3, stride=2, padding=1, bias=False)
            groups = gcd(8, spec.width)
            tied = spec.family in _RECURRENT_FAMILIES
            branch_count = 3 if spec.family == "H" else (1 if tied else spec.depth)
            self.branches = nn.ModuleList(
                [RecurrentBranch(spec.width, groups) for _ in range(branch_count)]
            )
            # Register the projection after the common C body: same-seed common
            # tensors retain their historical initialization stream.
            self.recall_projection = (
                nn.Conv2d(2 * spec.width, spec.width, kernel_size=1, bias=False)
                if self.recall_enabled else None
            )
            self.head = nn.Linear(spec.width, spec.num_classes, bias=True)
            self._initialize(seed)

    def _initialize(self, seed: int | None) -> None:
        # Draw stem/head first so their values do not depend on branch count or sharing.
        generator = torch.Generator(device="cpu")
        if seed is None:
            seed = torch.seed()
        generator.manual_seed(seed)
        for module in (self.stem, self.head):
            _reset_with_generator(module, generator)
        for branch in self.branches:
            for module in branch.modules():
                _reset_with_generator(module, generator)
        if self.recall_projection is not None:
            _reset_with_generator(self.recall_projection, generator)

    def encode(self, x: Tensor) -> Tensor:
        """Map an input to the constant-resolution state used by recurrence."""
        return self.stem(x)

    def initial_state(self, encoded: Tensor) -> Tensor:
        """Return h0; it aliases the differentiable encoded input by design."""
        return encoded

    def step(
        self, h: Tensor, encoded: Tensor | None = None, *, iteration: int | None = None
    ) -> Tensor:
        """Apply one model-owned update; untied families require a zero-based block index."""
        if self.spec.family == "H":
            if iteration is None or type(iteration) is not int or not 0 <= iteration < 8:
                raise ValueError("family H step requires iteration in 0..7")
            branch_index = 0 if iteration == 0 else (2 if iteration == 7 else 1)
            return h + self.spec.alpha * self.branches[branch_index](h)
        if self.spec.family not in _RECURRENT_FAMILIES:
            if iteration is None or type(iteration) is not int or not 0 <= iteration < self.spec.depth:
                raise ValueError("untied step requires iteration in the trained depth")
            return h + self.spec.alpha * self.branches[iteration](h)
        branch_input = h
        if self.recall_enabled:
            if encoded is None:
                raise ValueError("encoded state is required when recall is enabled")
            branch_input = self.recall_projection(torch.cat((h, encoded), dim=1))
        return h + self.spec.alpha * self.branches[0](branch_input)

    def readout(self, h: Tensor) -> Tensor:
        """Classify a recurrent state without applying another update."""
        return self.head(F.adaptive_avg_pool2d(h, 1).flatten(1))

    @staticmethod
    def detached_state_norm(h: Tensor) -> Tensor:
        """Return per-example L2 norms without retaining an autograd graph."""
        return h.detach().flatten(1).norm(dim=1)

    def forward(self, x: Tensor, *, capture_hidden_norms: bool = False, inference_k: int | None = None):
        k = self.spec.depth if inference_k is None else inference_k
        if type(k) is not int or k < 1:
            raise ValueError("inference_k must be a positive integer")
        if self.spec.family == "H" and k != 8:
            raise ValueError("family H supports inference_k=8 only")
        if self.spec.family not in _RECURRENT_FAMILIES and k > self.spec.depth:
            raise ValueError("inference_k cannot exceed the fixed model depth")
        encoded = self.encode(x)
        h = self.initial_state(encoded)
        norms = [] if capture_hidden_norms else None
        for index in range(k):
            h = self.step(h, encoded, iteration=index)
            if norms is not None:
                norms.append(self.detached_state_norm(h))
        logits = self.readout(h)
        if norms is None:
            return logits
        return logits, torch.stack(norms, dim=1)


class RecurrentBranch(nn.Module):
    def __init__(self, width: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, width, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, width, affine=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.relu2(self.norm2(self.conv1(self.relu1(self.norm1(x))))))


def _reset_with_generator(module: nn.Module, generator: torch.Generator) -> None:
    if isinstance(module, nn.Conv2d):
        _uniform_parameter(module.weight, module.weight.shape[1] * module.kernel_size[0] * module.kernel_size[1], generator)
        if module.bias is not None:
            fan_in = module.weight.shape[1] * module.kernel_size[0] * module.kernel_size[1]
            _uniform_parameter(module.bias, fan_in, generator)
    elif isinstance(module, nn.Linear):
        _uniform_parameter(module.weight, module.in_features, generator)
        if module.bias is not None:
            _uniform_parameter(module.bias, module.in_features, generator)
    elif isinstance(module, nn.GroupNorm):
        with torch.no_grad():
            module.weight.fill_(1)
            module.bias.zero_()


def _uniform_parameter(parameter: Tensor, fan_in: int, generator: torch.Generator) -> None:
    bound = fan_in**-0.5
    values = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu").uniform_(
        -bound, bound, generator=generator
    )
    with torch.no_grad():
        parameter.copy_(values)


def build_model(spec: ModelSpec, *, seed: int | None = None) -> RecurrentCNN:
    """Construct a model; ``seed`` controls deterministic initialization without global RNG draws."""
    return RecurrentCNN(spec, seed=seed)
