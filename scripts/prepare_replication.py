#!/usr/bin/env python3
"""Prepare an inert, fresh resource root for the existing native operator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "source-manifest.json"
PROTOCOL = Path("benchmarks/minimal-generalization-20261015/protocol.json")
ENV_PATHS = (
    "HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "APPDATA", "LOCALAPPDATA",
    "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH",
    "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR",
)
DATA_MD5 = {"train": "16019d7e3df5f24257cddd939b257f8d",
            "meta": "7973b15100ade9c7d40fb424638fde48"}


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _copy_source_snapshot(source_root: Path, destination: Path) -> dict:
    manifest = json.loads((source_root / "provenance/source-manifest.json").read_text(encoding="utf-8"))
    originals = manifest.get("original_source_sha256")
    if not isinstance(originals, dict) or not originals:
        raise ValueError("source manifest lacks original_source_sha256")
    snapshot = destination / "source"
    snapshot.mkdir()
    hashes = {}
    for relative in sorted(originals):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe source manifest path: {relative}")
        source_file = source_root / rel
        if not source_file.is_file() or source_file.is_symlink():
            raise FileNotFoundError(f"source snapshot file is missing or not regular: {relative}")
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        hashes[relative] = _sha256(target)
        if _sha256(source_file) != hashes[relative]:
            raise OSError(f"source snapshot verification failed: {relative}")
    return {"schema": "public-replication-source-hashes-v1", "source_head": _head(source_root),
            "files": hashes, "original_source_sha256": originals}


def _head(source_root: Path) -> str:
    return subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def _copy_training_data(dataset_root: Path, destination: Path) -> dict:
    candidates = [dataset_root]
    if dataset_root.name != "cifar-100-python":
        candidates.append(dataset_root / "cifar-100-python")
    origin = next((path for path in candidates if path.is_dir()
                   and all((path / name).is_file() for name in ("train", "meta"))), None)
    if origin is None:
        raise FileNotFoundError("expected local CIFAR-100 directory or its cifar-100-python child")
    for name in ("train", "meta"):
        path = origin / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"required regular CIFAR-100 {name} file is missing")
        if _md5(path) != DATA_MD5[name]:
            raise ValueError(f"CIFAR-100 {name} checksum is not the supported official file")
    target = destination / "data" / "cifar-100-python"
    target.mkdir(parents=True)
    copied = {}
    for name in ("train", "meta"):
        source = origin / name
        target_file = target / name
        shutil.copyfile(source, target_file)
        if _md5(target_file) != DATA_MD5[name] or _sha256(target_file) != _sha256(source):
            raise OSError(f"copied CIFAR-100 {name} failed integrity verification")
        copied[name] = {"md5": DATA_MD5[name], "sha256": _sha256(target_file)}
    return {"dataset": "CIFAR-100 official training partition", "provenance": "local user-supplied files; checksum-verified",
            "files": copied, "excluded": ["test"], "input": str(origin.resolve())}


def prepare(root: Path, training_root: Path) -> dict:
    root = root.expanduser().absolute()
    if not root.parent.is_dir():
        raise FileNotFoundError(f"root parent must already exist: {root.parent}")
    root = root.parent.resolve() / root.name
    training_root = training_root.expanduser().resolve(strict=True)
    if root == training_root or training_root in root.parents or root in training_root.parents:
        raise ValueError("replication root and read-only dataset input must not overlap")
    root.mkdir(exist_ok=False)
    if any(root.iterdir()):
        raise RuntimeError("newly created root is unexpectedly nonempty")

    source_hashes = _copy_source_snapshot(ROOT, root)
    head = source_hashes["source_head"]
    if len(head) != 40:
        raise ValueError("git HEAD must be a full commit hash")
    (root / "source_hashes.json").write_bytes(_json_bytes(source_hashes))
    protocol_file = root / "source" / PROTOCOL
    if not protocol_file.is_file():
        raise FileNotFoundError("protocol is absent from immutable source snapshot")
    protocol_target = root / "protocol.json"
    shutil.copyfile(protocol_file, protocol_target)
    if protocol_target.read_bytes() != protocol_file.read_bytes():
        raise OSError("protocol snapshot copy failed verification")

    data_provenance = _copy_training_data(training_root, root)
    proof = {
        "schema": "public-reproduction-resource-proof-v1", "classification": "DISPOSABLE_VERIFIED",
        "basis": "exclusive root directory created by mkdir(exist_ok=False) and verified empty before writes",
        "root": str(root.resolve()), "source_head": head,
        "owned_paths": ["source", "source_hashes.json", "protocol.json", "data", "study", "probe", "evidence", "private"],
        "readonly_inputs": ["source", "source_hashes.json", "protocol.json",
                            "data/cifar-100-python/train", "data/cifar-100-python/meta"],
        "excluded_inputs": ["data/cifar-100-python/test"],
        "stateful_resources": {"filesystem_root": str(root.resolve()),
                               "classification": "DISPOSABLE_VERIFIED"},
        "network": "not used", "database": "none", "external_services": "none",
    }
    _exclusive_json(root / "resource-proof.json", proof)
    private_dirs = " ".join(f'"$ROOT/private/{key.lower()}"' for key in ENV_PATHS)
    env = " ".join(f'{key}="$ROOT/private/{key.lower()}"' for key in ENV_PATHS)
    common = (f'ROOT="{root}"; mkdir -p {private_dirs}; export {env} '
              'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 '
              'MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12')
    operator = 'python benchmarks/minimal-generalization-20261015/run_study.py'
    evaluate = f'{operator} --phase evaluate --root "$ROOT" --source-head {head}'
    gated_test_copy = (
        "MANUAL AND GATED ONLY after all 15 runs complete and checkpoint/artifact freeze is independently verified: "
        f'test ! -e "$ROOT/data/cifar-100-python/test" && cp -- {shlex.quote(str(Path(data_provenance["input"]) / "test"))} '
        '"$ROOT/data/cifar-100-python/test"\n# Then, only after evaluation is separately authorized:\n' + evaluate
    )
    return {
        "status": "prepared", "root": str(root), "source_head": head,
        "source_files_verified": len(source_hashes["files"]), "protocol_copy_verified": True,
        "training_data": data_provenance, "full_replication": "NOT RUN",
        "independent_reanalysis": "requires only the published JSON results",
        "commands": {
            "environment_prefix": common,
            "train": f'{operator} --phase train --root "$ROOT" --source-head {head}',
            "evaluate": evaluate,
            "postfreeze_test_copy": gated_test_copy,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="new, previously nonexistent run root")
    parser.add_argument("--training-root", required=True, type=Path,
                        help="read-only local CIFAR-100 root or cifar-100-python directory")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.root, args.training_root), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
