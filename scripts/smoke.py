#!/usr/bin/env python3
"""Run a tiny, synthetic CPU-only check using private temporary storage."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATHS = (
    "HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "APPDATA", "LOCALAPPDATA",
    "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH",
    "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR",
)


def _private_environment(root: Path) -> dict[str, str]:
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ}
    cache = root / "private"
    for key in PRIVATE_PATHS:
        path = cache / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update({
        "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
    })
    return env


def _run(env: dict[str, str], *args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "recurrent_cnn.cli", *args], cwd=ROOT,
        env=env, check=True, text=True, capture_output=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="recurrent-cnn-smoke-") as temporary:
        run_root = Path(temporary).resolve()
        env = _private_environment(run_root)
        data_root = run_root / "data-root"
        data_root.mkdir()
        prepared = run_root / "prepared"
        run_dir = run_root / "run"
        _run(env, "prepare", "--root", str(data_root), "--output", str(prepared),
             "--synthetic", "--per-class", "3", "--paired-seed", "0")
        result = _run(env, "smoke", "--config", str(ROOT / "configs" / "smoke.yaml"),
                      "--data", str(prepared), "--run-dir", str(run_dir), "--steps", "2")
        receipt = result.get("result", {})
        if receipt.get("status") != "complete" or receipt.get("update") != 2:
            raise RuntimeError("native synthetic smoke did not complete exactly two updates")
        print(json.dumps({"scientific": False, "status": "complete", "root": str(run_root),
                          "updates": receipt["update"], "namespace": "synthetic"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
