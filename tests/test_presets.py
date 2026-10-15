from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from recurrent_cnn import cli, data, engine, profiling, protocol
from recurrent_cnn.config import TrainConfig


ROOT = Path(__file__).parents[1]
PRESETS = {
    "smoke.yaml", "overfit.yaml", "pilot.yaml", "train_e_full.yaml",
    "matrix_core.yaml", "matrix_full.yaml", "profile_rtx5090.yaml",
}


def _config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_every_preset_parses_strictly_without_execution_or_acquisition(tmp_path, monkeypatch):
    configs = ROOT / "configs"
    assert {path.name for path in configs.iterdir()} == PRESETS
    forbidden = []
    for owner, name in ((data, "prepare_data"), (protocol, "_open_official_test"),
                        (protocol, "_open_cifar100_c"), (engine, "train_run"),
                        (profiling, "profile_model")):
        monkeypatch.setattr(owner, name, lambda *a, _name=name, **k: forbidden.append(_name))

    parsed = {}
    for name in sorted(PRESETS):
        path = configs / name
        if name in {"smoke.yaml", "overfit.yaml", "pilot.yaml", "train_e_full.yaml"}:
            parsed[name] = TrainConfig.from_yaml(path.read_text(encoding="utf-8"))
        elif name.startswith("matrix_"):
            parsed[name] = cli._matrix_config(path, tmp_path / "explicit-root", None)
        else:
            parsed[name] = cli.resolve_profile_request(path)
    assert forbidden == []

    smoke = parsed["smoke.yaml"]
    assert (smoke.family, smoke.depth, smoke.width, smoke.role, smoke.synthetic) == ("C", 2, 4, "smoke", True)
    assert (smoke.total_updates, smoke.batch_size, smoke.validation_cadence) == (2, 2, 1)
    assert (smoke.split_seed, smoke.paired_seed, smoke.fraction, smoke.device, smoke.amp) == (1729, 0, 1.0, "cpu", False)

    overfit = parsed["overfit.yaml"]
    assert (overfit.family, overfit.depth, overfit.width, overfit.role, overfit.synthetic) == ("C", 2, 4, "overfit", True)
    assert (overfit.total_updates, overfit.batch_size, overfit.validation_cadence) == (12, 16, 4)
    assert (overfit.split_seed, overfit.paired_seed, overfit.device, overfit.amp) == (1729, 0, "cpu", False)

    pilot = parsed["pilot.yaml"]
    assert (pilot.family, pilot.depth, pilot.width, pilot.alpha, pilot.fraction) == ("E", 8, 32, 0.1, 0.1)
    assert (pilot.paired_seed, pilot.split_seed, pilot.role, pilot.synthetic) == (101, 1729, "pilot", False)
    assert (pilot.total_updates, pilot.batch_size, pilot.validation_cadence, pilot.device, pilot.amp) == (200, 128, 100, "cuda", False)
    assert (pilot.learning_rate, pilot.progressive_weight) == (0.01, 0.5)
    assert (pilot.recall, pilot.variable_depth, pilot.progressive_loss) == (True, True, True)

    research = parsed["train_e_full.yaml"]
    assert (research.family, research.depth, research.width, research.fraction) == ("E", 8, 32, 1.0)
    assert (research.role, research.synthetic, research.paired_seed, research.split_seed) == ("research", False, 11, 1729)
    assert (research.total_updates, research.cosine_horizon, research.learning_rate) == (2000, 2000, 0.01)
    assert (research.batch_size, research.validation_cadence, research.device, research.amp) == (128, 100, "cuda", False)
    assert (research.progressive_weight, research.recall, research.variable_depth, research.progressive_loss) == (0.5, True, True, True)

    for name, full, expected_count in (("matrix_core.yaml", False, 230), ("matrix_full.yaml", True, 290)):
        request = parsed[name]
        assert request == {"protocol_version": 2, "full": full, "batch_size": 128,
                           "device": "cuda", "amp": False, "seeds": [11, 23, 37, 53, 71],
                           "root": str(tmp_path / "explicit-root")}
        # This is canonical roster metadata only; it does not build or execute runs.
        assert protocol.amendment_definition()["roster"]["core" if not full else "full"] == expected_count
        assert "manifest" not in _config(name) and "updates" not in _config(name)

    profile = parsed["profile_rtx5090.yaml"]
    assert (profile["family"], profile["depth"], profile["width"], profile["alpha"]) == ("E", 8, 32, 0.1)
    assert (profile["device"], profile["amp"], profile["warmup"], profile["repeats"], profile["batch_size"]) == ("cuda", False, 10, 50, 128)
    assert profile["inference_grid"] == (1, 2, 4, 8, 12, 16, 24, 32, 64, 128)
    assert profile["sweep_examples"] == 128 and profile["progressive_weight"] == 0.5
    assert (profile["recall"], profile["variable_depth"], profile["progressive_loss"]) == (True, True, True)
    assert "output" not in _config("profile_rtx5090.yaml")


def test_all_preset_parsers_reject_unknown_keys(tmp_path):
    unknown = {"unexpected_typo": True}
    with pytest.raises(ValueError, match="unknown configuration keys"):
        TrainConfig.from_yaml(yaml.safe_dump({"family": "C", "depth": 2, **unknown}))
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"protocol_version": 2, **unknown}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown matrix configuration keys"):
        cli._matrix_config(matrix, tmp_path / "root", None)
    with pytest.raises(ValueError, match="unknown profile config"):
        cli.resolve_profile_request({"family": "E", "depth": 8, **unknown})


def test_only_smoke_preset_executes_in_generated_cpu_scratch(tmp_path, capsys):
    data_dir = tmp_path / "generated-arrays"
    run_dir = tmp_path / "smoke-run"
    assert cli.main(["prepare", "--root", str(tmp_path / "synthetic-root"), "--output", str(data_dir),
                     "--synthetic", "--per-class", "3", "--split-seed", "1729", "--paired-seed", "0"]) == 0
    capsys.readouterr()
    assert cli.main(["smoke", "--config", str(ROOT / "configs" / "smoke.yaml"), "--data", str(data_dir),
                     "--run-dir", str(run_dir)]) == 0
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["result"]["status"] == "complete"
    assert outcome["result"]["update"] == 2
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["config"]["role"] == "smoke"
    assert record["config"]["device"] == "cpu" and record["config"]["synthetic"] is True
    assert (run_dir / "last.pt").is_file() and (run_dir / "best.pt").is_file()
