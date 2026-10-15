"""Safe public command-line adapters for the recurrent CNN study."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

from . import data, engine, profiling, protocol
from .config import TrainConfig
from .data import CifarArrays
from .models import ModelSpec


_PREPARED_SCHEMA = "recurrent-cnn-prepared-arrays-v1"
_MATRIX_KEYS = {"root", "manifest", "protocol_version", "updates", "batch_size", "device", "amp",
                "full", "seeds", "fractions", "roster", "split_seed", "paired_seeds", "pilot_seeds",
                "pilot_updates", "learning_rates", "data_hash"}
_PROFILE_KEYS = {"family", "depth", "width", "alpha", "num_classes", "device", "amp", "warmup",
                 "repeats", "batch_size", "inference_grid", "sweep_examples", "progressive_weight",
                 "recall", "variable_depth", "progressive_loss"}
_PROFILE_DEFAULTS = {"width": 1, "alpha": 0.1, "num_classes": 100, "device": "cpu", "amp": False,
                     "warmup": 1, "repeats": 1, "batch_size": 1, "inference_grid": None,
                     "sweep_examples": 1, "progressive_weight": 0.5, "recall": None,
                     "variable_depth": None, "progressive_loss": None}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON: {path}") from exc


def _new_directory(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if target.exists():
        raise FileExistsError(f"destination already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"destination parent does not exist: {target.parent}")
    return target


def _exclusive_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data.canonical_json(value) + "\n")


def _array_identity(arrays: CifarArrays) -> str:
    return engine._array_identity(arrays)


def _save_arrays(arrays: CifarArrays, output: str | Path, *, manifest: dict[str, Any] | None = None) -> Path:
    target = _new_directory(output)
    target.mkdir()
    np.savez(target / "arrays.npz", images=arrays.images, labels=arrays.labels)
    metadata = {"schema": _PREPARED_SCHEMA, "namespace": arrays.namespace,
                "metadata": arrays.metadata or {}, "array_identity": _array_identity(arrays)}
    _exclusive_json(target / "metadata.json", metadata)
    if manifest is not None:
        data.save_manifest(target / "manifest.json", manifest)
    return target


def _load_arrays(source: str | Path, *, require_training: bool = False) -> tuple[CifarArrays, dict[str, Any] | None]:
    path = Path(source).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"prepared-data directory does not exist: {path}")
    try:
        with np.load(path / "arrays.npz", allow_pickle=False) as stored:
            if set(stored.files) != {"images", "labels"}:
                raise ValueError("prepared arrays must contain only images and labels")
            images, labels = stored["images"], stored["labels"]
    except (OSError, ValueError) as exc:
        raise ValueError("cannot load prepared arrays") from exc
    metadata = _json(path / "metadata.json")
    if not isinstance(metadata, dict) or metadata.get("schema") != _PREPARED_SCHEMA:
        raise ValueError("unsupported prepared-data metadata schema")
    arrays = CifarArrays(images, labels, metadata.get("namespace"), metadata.get("metadata"))
    if metadata.get("array_identity") != _array_identity(arrays):
        raise ValueError("prepared array identity mismatch")
    manifest_path = path / "manifest.json"
    manifest = data.load_manifest(manifest_path) if manifest_path.is_file() else None
    if require_training:
        if manifest is None or arrays.namespace not in ("synthetic", "train"):
            raise ValueError("this action requires prepared training arrays and a canonical manifest")
        expected_synthetic = arrays.namespace == "synthetic"
        if manifest["synthetic"] != expected_synthetic:
            raise ValueError("prepared manifest population does not match arrays")
        max_index = max(manifest["indices"]["train"] + manifest["indices"]["validation"])
        if max_index >= len(arrays.labels):
            raise ValueError("prepared manifest indices exceed arrays")
    return arrays, manifest


def _training_population(arrays: CifarArrays, manifest: dict[str, Any], fraction: float,
                         config: TrainConfig) -> tuple[TrainConfig, np.ndarray, np.ndarray]:
    if manifest["paired_seed"] != config.paired_seed or manifest["split_seed"] != config.split_seed:
        raise ValueError("training config seed identities conflict with prepared manifest")
    key = f"{fraction:.2f}"
    try:
        train = np.asarray(manifest["indices"]["subsets"][key], dtype=np.int64)
        validation = np.asarray(manifest["indices"]["validation"], dtype=np.int64)
    except (KeyError, TypeError) as exc:
        raise ValueError("requested training fraction is absent from manifest") from exc
    split_hash, subset_hash = engine.training_identity_hashes(len(arrays.labels), train, validation)
    identities = {"data_hash": _array_identity(arrays), "split_hash": split_hash, "subset_hash": subset_hash}
    for field, actual in identities.items():
        supplied = getattr(config, field)
        if supplied and supplied != actual:
            raise ValueError(f"training config {field} conflicts with prepared population")
    resolved = replace(config, **identities)
    return resolved, train, validation


def _read_train_config(path: str | Path, role: str) -> TrainConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file does not exist: {config_path}")
    config = TrainConfig.from_yaml(config_path.read_text(encoding="utf-8"))
    if config.role != role:
        raise ValueError(f"{role} requires config role={role!r}")
    return config


def _cmd_prepare(args: argparse.Namespace) -> dict[str, Any]:
    _new_directory(args.output)
    if args.download and not args.official:
        raise ValueError("--download is allowed only with explicit --official")
    if (args.synthetic_heldout or args.synthetic_corruption_from) and args.official:
        raise ValueError("synthetic preparation modes cannot be combined with --official")
    if args.synthetic_heldout and args.synthetic_corruption_from:
        raise ValueError("heldout generation and corruption transformation are mutually exclusive")
    if args.synthetic_corruption_from:
        source, manifest = _load_arrays(args.synthetic_corruption_from)
        if source.namespace != "synthetic-test" or source.metadata.get("synthetic") is not True:
            raise ValueError("corruption source must be explicitly generated synthetic heldout data")
        if args.corruption not in {"gaussian_blur", "saturate", "spatter"}:
            raise ValueError("corruption must be gaussian_blur, saturate, or spatter")
        if args.severity is None or args.seed is None:
            raise ValueError("corruption generation requires explicit --severity and --seed")
        if args.severity not in range(1, 6) or args.seed < 0:
            raise ValueError("synthetic corruption severity must be 1..5 and seed nonnegative")
        images = data._validation_transform(source.images, args.corruption, args.severity,
                                           np.random.default_rng(args.seed))
        arrays = CifarArrays(images, source.labels.copy(), "synthetic-test-corruption", {
            "source": "synthetic-heldout-corruption", "synthetic": True,
            "corruption": args.corruption, "severity": args.severity, "seed": args.seed,
            "origin_identity": _array_identity(source)})
        target = _save_arrays(arrays, args.output)
    elif args.synthetic_heldout:
        if args.per_class < 2:
            raise ValueError("--per-class must be at least 2")
        arrays = data.prepare_data(args.root, synthetic=True, synthetic_per_class=args.per_class, seed=args.seed or 0)
        arrays = CifarArrays(arrays.images, arrays.labels, "synthetic-test", {
            "source": "synthetic-heldout", "synthetic": True, "seed": args.seed or 0})
        target = _save_arrays(arrays, args.output)
    else:
        if args.synthetic == args.official:
            raise ValueError("choose exactly one of --synthetic or --official")
        if args.per_class < 2 and args.synthetic:
            raise ValueError("--per-class must be at least 2")
        arrays = data.prepare_data(args.root, download=args.download, synthetic=args.synthetic,
                                   synthetic_per_class=args.per_class, seed=args.seed or 0)
        synthetic = arrays.namespace == "synthetic"
        train, validation = data.make_split(arrays.labels, args.split_seed, synthetic=synthetic)
        subsets = data.make_subsets(arrays.labels, train, args.paired_seed, synthetic=synthetic)
        manifest = data.build_manifest(split_seed=args.split_seed, paired_seed=args.paired_seed,
                                       train_indices=train, validation_indices=validation,
                                       subsets=subsets, synthetic=synthetic, train_namespace=arrays.namespace)
        target = _save_arrays(arrays, args.output, manifest=manifest)
    return {"prepared_data": str(target), "namespace": arrays.namespace,
            "array_identity": _array_identity(arrays), "manifest": str(target / "manifest.json") if (target / "manifest.json").exists() else None}


def _cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    role = args.command
    expected_role = "research" if role == "train" else role
    config = _read_train_config(args.config, expected_role)
    arrays, manifest = _load_arrays(args.data, require_training=True)
    if arrays.namespace != ("synthetic" if config.synthetic else "train"):
        raise ValueError("config synthetic mode does not match prepared arrays")
    resolved, train, validation = _training_population(arrays, manifest, config.fraction, config)
    if role == "overfit":
        if not config.synthetic or len(train) < config.batch_size:
            raise ValueError("overfit requires a synthetic population containing one complete batch")
        train = train[:config.batch_size]
        split_hash, subset_hash = engine.training_identity_hashes(len(arrays.labels), train, validation)
        for name, value in (("split_hash", split_hash), ("subset_hash", subset_hash)):
            supplied = getattr(config, name)
            if supplied and supplied != value:
                raise ValueError(f"training config {name} conflicts with one-batch overfit population")
        resolved = replace(resolved, split_hash=split_hash, subset_hash=subset_hash)
    _new_directory(args.run_dir)
    result = engine.train_run(resolved, arrays, train, validation, args.run_dir, step_limit=args.steps)
    return {"run_dir": str(args.run_dir), "config": resolved.to_dict(), "result": result}


def _matrix_config(path: str | Path, root: str | Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    if path:
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"matrix config does not exist: {config_path}")
        try:
            value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError("invalid matrix YAML") from exc
        if not isinstance(value, Mapping):
            raise ValueError("matrix YAML must be a mapping")
        unknown = set(value) - _MATRIX_KEYS
        if unknown:
            raise ValueError(f"unknown matrix configuration keys: {sorted(unknown)}")
        config = dict(value)
        if "root" in config and Path(config["root"]).expanduser() != Path(root).expanduser():
            raise ValueError("matrix YAML root conflicts with explicit --root")
        config["root"] = str(root)
        if manifest is not None:
            if "manifest" in config and config["manifest"] != manifest:
                raise ValueError("matrix YAML manifest conflicts with prepared-data manifest")
            config["manifest"] = manifest
        return config
    if manifest is None:
        raise ValueError("matrix planning requires a prepared manifest or explicit matrix config")
    return {"root": str(root), "manifest": manifest}


def _cmd_matrix(args: argparse.Namespace) -> dict[str, Any]:
    if args.action == "plan":
        _, manifest = _load_arrays(args.data, require_training=True) if args.data else (None, None)
        config = _matrix_config(args.config, args.root, manifest)
        if args.amendment:
            config.update(protocol_version=2, updates=2, batch_size=args.batch_size or config.get("batch_size", 2),
                          device="cpu", amp=False)
            phase = "smoke"
        else:
            phase = "smoke" if args.phase == "smoke" else "research"
        return protocol.build_matrix(config, phase=phase)
    if args.action == "pilot":
        arrays, _ = _load_arrays(args.data, require_training=True)
        if not args.execute:
            raise ValueError("matrix pilot requires deliberate --execute")
        return {"pilots": protocol.execute_pilots(args.root, arrays, execute=True)}
    if args.action == "preregister":
        return protocol.preregister_protocol(args.root)
    if args.action == "execute":
        arrays, _ = _load_arrays(args.data, require_training=True)
        if not args.execute:
            raise ValueError("matrix execute requires deliberate --execute")
        return {"runs": protocol.execute_matrix(args.matrix or Path(args.root) / "matrix.json", arrays,
                                                 args.root, execute=True, cohort=args.cohort,
                                                 approved_phase=args.approved_phase)}
    if args.action == "seal":
        arrays, _ = _load_arrays(args.data, require_training=True) if args.data else (None, None)
        return protocol.freeze_protocol(args.root, cohort=args.cohort, validation_arrays=arrays)
    raise ValueError(f"unsupported matrix action: {args.action}")


def _parse_corruption(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("corruption must be NAME=PREPARED_DATA_DIRECTORY")
    if any(not (character.isalnum() or character == "_") for character in name) or "cifar100c" in name:
        raise argparse.ArgumentTypeError("synthetic corruption names use alphanumeric/underscore only")
    return name, Path(path)


def _cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.synthetic_heldout:
        if args.official_data_root or args.cifar100c_root or args.download_test:
            raise ValueError("synthetic and official evaluation inputs are mutually exclusive")
        arrays, _ = _load_arrays(args.synthetic_heldout)
        if arrays.namespace != "synthetic-test" or arrays.metadata.get("synthetic") is not True:
            raise ValueError("evaluation requires explicitly generated synthetic-test arrays")
        identities = [_array_identity(arrays)]
        corruptions: dict[str, CifarArrays] = {}
        for name, path in args.synthetic_corruption:
            if name in corruptions:
                raise ValueError(f"duplicate synthetic corruption name: {name}")
            corrupted, _ = _load_arrays(path)
            if (corrupted.namespace != "synthetic-test-corruption"
                    or corrupted.metadata.get("source") != "synthetic-heldout-corruption"
                    or corrupted.metadata.get("synthetic") is not True):
                raise ValueError(f"{name} is not a generated synthetic heldout corruption")
            if not np.array_equal(corrupted.labels, arrays.labels) or _array_identity(corrupted) in identities:
                raise ValueError("synthetic corruption labels must match and images must have distinct identities")
            identities.append(_array_identity(corrupted))
            corruptions[name] = corrupted
        if len(corruptions) < 2:
            raise ValueError("synthetic evaluation requires at least two generated corruptions")
        indices = np.arange(len(arrays.labels), dtype=np.int64) if args.indices is None else np.asarray(args.indices, dtype=np.int64)
        if len(indices) == 0 or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(arrays.labels)):
            raise ValueError("synthetic indices must be a nonempty unique in-range selection")
        return protocol.evaluate_frozen(args.root, args.run_id, synthetic_arrays=arrays,
                                        synthetic_indices=indices, synthetic_corruptions=corruptions,
                                        device=args.device, cohort=args.cohort)
    if args.synthetic_corruption or args.indices is not None:
        raise ValueError("synthetic-only options require --synthetic-heldout")
    if not args.official_data_root or not args.cifar100c_root:
        raise ValueError("official evaluation requires explicit --official-data-root and --cifar100c-root")
    return protocol.evaluate_frozen(args.root, args.run_id, data_root=args.official_data_root,
                                    cifar100c_root=args.cifar100c_root, download_test=args.download_test,
                                    device=args.device, cohort=args.cohort)


def _cmd_analyze(args: argparse.Namespace) -> dict[str, Any]:
    from . import analysis

    return analysis.aggregate_runs(args.run_table, args.output, include_synthetic=args.include_synthetic,
                                   cohort=args.cohort)


def resolve_profile_request(config: Mapping[str, Any] | str | Path | None,
                            explicit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve strict profile YAML and CLI overrides without probing a backend."""
    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"profile config does not exist: {config_path}")
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError("invalid profile YAML") from exc
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("profile config must be a mapping")
    for name, values in (("profile config", config), ("profile overrides", explicit or {})):
        if not isinstance(values, Mapping) or any(not isinstance(key, str) for key in values):
            raise ValueError(f"{name} keys must be strings")
        unknown = set(values) - _PROFILE_KEYS
        if unknown:
            raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
    request = dict(_PROFILE_DEFAULTS)
    request.update(config)
    request.update({key: value for key, value in (explicit or {}).items() if value is not None})
    if request.get("family") is None or request.get("depth") is None:
        raise ValueError("profile family and depth are required in config or CLI arguments")
    if request["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("profile device must be cpu, cuda, or mps")
    if type(request["amp"]) is not bool:
        raise ValueError("profile amp must be a boolean")
    spec = ModelSpec(request["family"], request["depth"], request["width"], request["alpha"],
                     request["num_classes"], recall=request["recall"],
                     variable_depth=request["variable_depth"], progressive_loss=request["progressive_loss"])
    profiling._positive_int("warmup", request["warmup"], allow_zero=True)
    profiling._positive_int("repeats", request["repeats"])
    profiling._positive_int("batch_size", request["batch_size"])
    if request["sweep_examples"] is not None:
        profiling._positive_int("sweep_examples", request["sweep_examples"])
    weight = request["progressive_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0.5 <= weight <= 1:
        raise ValueError("progressive_weight must be finite and in [0.5, 1]")
    if spec.family not in ("E", "F") and weight != 0.5:
        raise ValueError("progressive_weight applies only to E/F")
    grid = request["inference_grid"]
    if grid is not None:
        if not isinstance(grid, (list, tuple)):
            raise ValueError("inference_grid must be a sequence of integers")
        profiling._validate_profile_grid(spec, grid)
        request["inference_grid"] = tuple(grid)
    return request


def _profile_explicit(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in _PROFILE_KEYS if getattr(args, key, None) is not None}


def _cmd_profile(args: argparse.Namespace) -> dict[str, Any]:
    request = resolve_profile_request(args.config, _profile_explicit(args))
    spec = ModelSpec(request["family"], request["depth"], request["width"], request["alpha"],
                     request["num_classes"], recall=request["recall"],
                     variable_depth=request["variable_depth"], progressive_loss=request["progressive_loss"])
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {args.output}")
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"profile output parent does not exist: {args.output.parent}")
    return profiling.profile_model(spec, request["device"], request["amp"], request["warmup"],
                                  request["repeats"], request["batch_size"], args.output,
                                  progressive_weight=request["progressive_weight"],
                                  inference_grid=request["inference_grid"],
                                  sweep_examples=request["sweep_examples"])


def _add_train_command(subparsers: Any, name: str) -> None:
    parser = subparsers.add_parser(name, help=f"run an explicit {name} training recipe")
    parser.add_argument("--config", required=True, help="strict TrainConfig YAML")
    parser.add_argument("--data", required=True, help="prepared-data directory")
    parser.add_argument("--run-dir", required=True, help="new exclusive run directory")
    parser.add_argument("--steps", type=int, help="bounded update count")
    parser.set_defaults(handler=_cmd_train)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recurrent-cnn", description="Explicit, guarded recurrent CNN experiment commands")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="create an explicit synthetic prepared-data artifact")
    prepare.add_argument("--root", required=True, help="explicit data root (synthetic generation does not acquire from it)")
    prepare.add_argument("--output", required=True, help="new prepared-data directory")
    prepare.add_argument("--synthetic", action="store_true", help="generate marked synthetic training arrays")
    prepare.add_argument("--official", action="store_true", help="load explicit local official training data")
    prepare.add_argument("--download", action="store_true", help="explicitly allow official training acquisition")
    prepare.add_argument("--synthetic-heldout", action="store_true", help="generate a separate synthetic-test population")
    prepare.add_argument("--synthetic-corruption-from", help="heldout artifact to transform synthetically")
    prepare.add_argument("--corruption", choices=("gaussian_blur", "saturate", "spatter"))
    prepare.add_argument("--severity", type=int)
    prepare.add_argument("--per-class", type=int, default=20)
    prepare.add_argument("--seed", type=int)
    prepare.add_argument("--split-seed", type=int, default=1729)
    prepare.add_argument("--paired-seed", type=int, default=0)
    prepare.set_defaults(handler=_cmd_prepare)
    _add_train_command(commands, "overfit")
    _add_train_command(commands, "smoke")
    _add_train_command(commands, "train")

    matrix = commands.add_parser("matrix", help="use the existing protocol matrix authority")
    actions = matrix.add_subparsers(dest="action", required=True)
    plan = actions.add_parser("plan", help="write an inert matrix plan")
    plan.add_argument("--root", required=True)
    plan.add_argument("--data", help="prepared training artifact with manifest")
    plan.add_argument("--config", help="strict matrix YAML for multi-manifest research")
    plan.add_argument("--amendment", action="store_true", help="route only to synthetic amended-v2 smoke")
    plan.add_argument("--phase", choices=("smoke", "research"), default="smoke")
    plan.add_argument("--batch-size", type=int)
    plan.set_defaults(handler=_cmd_matrix)
    for action_name in ("pilot", "preregister", "execute", "seal"):
        action_parser = actions.add_parser(action_name, help=f"run the existing matrix {action_name} authority")
        action_parser.add_argument("--root", required=True)
        if action_name in ("pilot", "execute", "seal"):
            action_parser.add_argument("--data", required=action_name in ("pilot", "execute"),
                                       help="prepared synthetic training arrays")
        if action_name in ("pilot", "execute"):
            action_parser.add_argument("--execute", action="store_true", help="deliberately authorize this action")
        if action_name == "execute":
            action_parser.add_argument("--matrix", help="explicit matrix JSON path")
            action_parser.add_argument("--cohort", choices=("phase_c", "core", "full", "smoke"))
            action_parser.add_argument("--approved-phase", choices=("phase_c", "phase_d"),
                                       help="separate operator approval marker; never implied by --execute")
        if action_name == "seal":
            action_parser.add_argument("--cohort", choices=("phase_c", "core", "full", "smoke"))
        action_parser.set_defaults(handler=_cmd_matrix, action=action_name)

    evaluate = commands.add_parser("evaluate", help="evaluate only through the frozen protocol authority")
    evaluate.add_argument("--root", required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--synthetic-heldout")
    evaluate.add_argument("--synthetic-corruption", action="append", type=_parse_corruption, default=[])
    evaluate.add_argument("--official-data-root", help="explicit official test-data root")
    evaluate.add_argument("--cifar100c-root", help="explicit official CIFAR-100-C root")
    evaluate.add_argument("--download-test", action="store_true", help="explicit official test-data acquisition")
    evaluate.add_argument("--indices", nargs="*", type=int)
    evaluate.add_argument("--cohort", choices=("phase_c", "core", "full", "smoke"))
    evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    evaluate.set_defaults(handler=_cmd_evaluate)

    analyze = commands.add_parser("analyze", help="aggregate actual result artifacts")
    analyze.add_argument("--run-table", required=True, help="actual run/result table or study root")
    analyze.add_argument("--output", required=True, help="new analysis output directory")
    analyze.add_argument("--include-synthetic", action="store_true")
    analyze.add_argument("--cohort")
    analyze.set_defaults(handler=_cmd_analyze)

    profile = commands.add_parser("profile", help="measure the actual CPU/GPU model objective on generated tensors")
    profile.add_argument("--config", help="strict profile request YAML; output remains an explicit CLI path")
    profile.add_argument("--family", choices=tuple("ABCDEF"))
    profile.add_argument("--depth", type=int)
    profile.add_argument("--width", type=int)
    profile.add_argument("--alpha", type=float)
    profile.add_argument("--num-classes", type=int)
    profile.add_argument("--device", choices=("cpu", "cuda", "mps"))
    profile.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    profile.add_argument("--warmup", type=int)
    profile.add_argument("--repeats", type=int)
    profile.add_argument("--batch-size", type=int)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--inference-grid", nargs="+", type=int)
    profile.add_argument("--sweep-examples", type=int)
    profile.add_argument("--progressive-weight", type=float)
    profile.add_argument("--recall", action=argparse.BooleanOptionalAction, default=None)
    profile.add_argument("--variable-depth", action=argparse.BooleanOptionalAction, default=None)
    profile.add_argument("--progressive-loss", action=argparse.BooleanOptionalAction, default=None)
    profile.set_defaults(handler=_cmd_profile)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
