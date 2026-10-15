from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pytest
import yaml

from recurrent_cnn import cli, protocol
from recurrent_cnn.config import TrainConfig


COMMANDS = ("prepare", "overfit", "smoke", "train", "matrix", "evaluate", "analyze", "profile")


def _call(arguments: list[str], capsys) -> dict:
    assert cli.main(arguments) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _write_config(path: Path, **overrides) -> Path:
    values = {"family": "C", "depth": 1, "width": 1, "role": "smoke", "synthetic": True,
              "total_updates": 1, "batch_size": 2, "validation_cadence": 1, "split_seed": 19}
    values.update(overrides)
    config = TrainConfig(**values)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=True), encoding="utf-8")
    return path


def test_public_help_and_entrypoints_are_inert(capsys, monkeypatch):
    opened = []
    monkeypatch.setattr(protocol, "_open_official_test", lambda *a, **k: opened.append("test"))
    monkeypatch.setattr(protocol, "_open_cifar100_c", lambda *a, **k: opened.append("c"))
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
           "CUDA_VISIBLE_DEVICES": ""}
    for command in COMMANDS:
        result = subprocess.run([sys.executable, "-m", "recurrent_cnn", command, "--help"],
                                text=True, capture_output=True, env=env, check=False,
                                stdin=subprocess.DEVNULL, timeout=30)
        assert result.returncode == 0, result.stderr
        assert command in result.stdout
    for action in ("plan", "pilot", "preregister", "execute", "seal"):
        result = subprocess.run([sys.executable, "-m", "recurrent_cnn", "matrix", action, "--help"],
                                text=True, capture_output=True, env=env, check=False,
                                stdin=subprocess.DEVNULL, timeout=30)
        assert result.returncode == 0, result.stderr
        assert action in result.stdout
    assert cli.main.__module__ == "recurrent_cnn.cli"
    package = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert package["project"]["scripts"]["recurrent-cnn"] == "recurrent_cnn.cli:main"
    with pytest.raises(SystemExit) as entrypoint_help:
        cli.main(["--help"])
    assert entrypoint_help.value.code == 0
    assert "recurrent-cnn" in capsys.readouterr().out
    assert opened == []


def test_help_and_profile_resolution_do_not_import_plotting(tmp_path):
    home = tmp_path / "home"
    mpl = tmp_path / "mpl"
    home.mkdir()
    mpl.mkdir()
    env = {**os.environ, "HOME": str(home), "MPLCONFIGDIR": str(mpl),
           "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    code = "\n".join([
        "import sys",
        "from recurrent_cnn import cli",
        "request = cli.resolve_profile_request({'family': 'E', 'depth': 8, 'device': 'cuda'})",
        "assert request['device'] == 'cuda'",
        "try:",
        "    cli.main(['--help'])",
        "except SystemExit as exc:",
        "    assert exc.code == 0",
        "assert 'matplotlib' not in sys.modules",
        "assert 'matplotlib.pyplot' not in sys.modules",
    ])
    result = subprocess.run([sys.executable, "-B", "-c", code], text=True,
                            capture_output=True, env=env, check=False,
                            stdin=subprocess.DEVNULL, timeout=30)
    assert result.returncode == 0, result.stderr
    assert list(mpl.iterdir()) == []
    assert not (home / ".matplotlib").exists()


def test_strict_config_and_prepared_array_identity_rejection(tmp_path, capsys):
    config = tmp_path / "bad.yaml"
    config.write_text("family: C\ndepth: 1\nunknown_typo: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration keys"):
        TrainConfig.from_yaml(config.read_text())
    rejected_run = tmp_path / "rejected-run"
    assert cli.main(["smoke", "--config", str(config), "--data", str(tmp_path / "absent"),
                     "--run-dir", str(rejected_run)]) == 1
    assert "unknown configuration keys" in capsys.readouterr().err
    assert not rejected_run.exists()
    artifact = tmp_path / "prepared"
    _call(["prepare", "--root", str(tmp_path / "data-root"), "--output", str(artifact),
           "--synthetic", "--per-class", "3", "--split-seed", "19"], capsys)
    assert cli.main(["prepare", "--root", str(tmp_path / "data-root"), "--output", str(artifact),
                     "--synthetic", "--per-class", "3", "--split-seed", "19"]) == 1
    capsys.readouterr()
    arrays_path = artifact / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as data:
        images, labels = data["images"].copy(), data["labels"].copy()
    images[0, 0, 0, 0] ^= np.uint8(1)
    np.savez(arrays_path, images=images, labels=labels)
    with pytest.raises(ValueError, match="identity mismatch"):
        cli._load_arrays(artifact)


def test_train_entrypoints_use_real_baseline_smoke_and_one_batch_overfit(tmp_path, capsys):
    artifact = tmp_path / "arrays"
    _call(["prepare", "--root", str(tmp_path / "root"), "--output", str(artifact),
           "--synthetic", "--per-class", "3", "--split-seed", "19"], capsys)
    smoke_config = _write_config(tmp_path / "smoke.yaml")
    smoke = _call(["smoke", "--config", str(smoke_config), "--data", str(artifact),
                   "--run-dir", str(tmp_path / "smoke-run")], capsys)
    assert smoke["result"]["status"] == "complete"
    assert smoke["result"]["update"] == 1
    overfit_config = _write_config(tmp_path / "overfit.yaml", role="overfit")
    overfit = _call(["overfit", "--config", str(overfit_config), "--data", str(artifact),
                     "--run-dir", str(tmp_path / "overfit-run")], capsys)
    assert overfit["result"]["status"] == "complete"
    assert overfit["result"]["update"] == 1
    assert json.loads((tmp_path / "smoke-run" / "run.json").read_text())["config"]["role"] == "smoke"
    assert json.loads((tmp_path / "overfit-run" / "run.json").read_text())["config"]["role"] == "overfit"


def test_public_train_maps_to_research_and_rejects_wrong_config_role(tmp_path, capsys, monkeypatch):
    artifact = tmp_path / "training"
    _call(["prepare", "--root", str(tmp_path / "root"), "--output", str(artifact),
           "--synthetic", "--per-class", "3", "--split-seed", "19"], capsys)
    wrong_role = _write_config(tmp_path / "wrong-role.yaml", role="smoke")
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(cli.engine, "train_run", lambda *a, **k: calls.append((a, k)))
        assert cli.main(["train", "--config", str(wrong_role), "--data", str(artifact),
                         "--run-dir", str(tmp_path / "wrong-role-run")]) == 1
        assert "research" in capsys.readouterr().err
    assert calls == []

    research_config = _write_config(tmp_path / "research.yaml", role="research")
    outcome = _call(["train", "--config", str(research_config), "--data", str(artifact),
                     "--run-dir", str(tmp_path / "research-run")], capsys)
    assert outcome["result"]["status"] == "complete" and outcome["result"]["update"] == 1
    record = json.loads((tmp_path / "research-run" / "run.json").read_text())
    assert record["config"]["role"] == "research"


def test_profile_yaml_resolution_is_strict_merged_and_backend_inert(tmp_path, monkeypatch, capsys):
    preset = tmp_path / "profile_rtx5090.yaml"
    preset.write_text(yaml.safe_dump({
        "family": "E", "depth": 8, "width": 2, "alpha": 0.1, "num_classes": 100,
        "device": "cuda", "amp": True, "warmup": 10, "repeats": 50, "batch_size": 128,
        "inference_grid": [1, 2, 4, 8, 12, 16, 24, 32, 64, 128], "sweep_examples": 4,
        "progressive_weight": 0.75, "recall": True, "variable_depth": True, "progressive_loss": True,
    }, sort_keys=True), encoding="utf-8")
    accessed = []
    monkeypatch.setattr(cli.profiling, "profile_model", lambda *a, **k: accessed.append("profile"))
    monkeypatch.setattr(cli.profiling, "resolve_backend", lambda *a, **k: accessed.append("backend"))

    base = cli._parser().parse_args(["profile", "--config", str(preset), "--output", str(tmp_path / "out.json")])
    request = cli.resolve_profile_request(base.config, cli._profile_explicit(base))
    assert request["device"] == "cuda" and request["family"] == "E" and request["depth"] == 8
    assert request["width"] == 2 and request["num_classes"] == 100
    assert request["warmup"] == 10 and request["repeats"] == 50 and request["amp"] is True
    assert request["progressive_weight"] == 0.75

    overridden = cli._parser().parse_args(["profile", "--config", str(preset), "--output",
                                           str(tmp_path / "out.json"), "--width", "3", "--device", "mps",
                                           "--no-amp"])
    resolved = cli.resolve_profile_request(overridden.config, cli._profile_explicit(overridden))
    assert resolved["width"] == 3 and resolved["device"] == "mps" and resolved["amp"] is False
    for device in ("cuda", "mps"):
        evaluation = cli._parser().parse_args(["evaluate", "--root", str(tmp_path / "study"),
            "--run-id", "r", "--official-data-root", str(tmp_path / "test"),
            "--cifar100c-root", str(tmp_path / "c"), "--device", device])
        assert evaluation.device == device
    assert accessed == []

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("family: E\ndepth: 8\nunknown_typo: true\n", encoding="utf-8")
    invalid_spec = tmp_path / "invalid-spec.yaml"
    invalid_spec.write_text("family: A\ndepth: 8\n", encoding="utf-8")
    for config_path in (unknown, invalid_spec):
        assert cli.main(["profile", "--config", str(config_path), "--output", str(tmp_path / "invalid.json")]) == 1
        assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "invalid.json").exists()
    assert accessed == []


def test_actual_public_amended_synthetic_lifecycle_and_artifacts(tmp_path, capsys):
    training = tmp_path / "training"
    _call(["prepare", "--root", str(tmp_path / "data-root"), "--output", str(training),
           "--synthetic", "--per-class", "3", "--split-seed", "19"], capsys)
    study = tmp_path / "study"
    _call(["matrix", "plan", "--root", str(study), "--data", str(training), "--amendment"], capsys)
    pilots = _call(["matrix", "pilot", "--root", str(study), "--data", str(training), "--execute"], capsys)
    assert len(pilots["pilots"]) == 6
    prereg = _call(["matrix", "preregister", "--root", str(study)], capsys)
    assert prereg["scientific"] is False
    runs = _call(["matrix", "execute", "--root", str(study), "--data", str(training), "--execute"], capsys)
    assert len(runs["runs"]) == 6
    freeze = _call(["matrix", "seal", "--root", str(study), "--data", str(training)], capsys)
    assert freeze["cohort"] == "smoke" and freeze["evidence_eligible"] is False

    heldout = tmp_path / "heldout"
    _call(["prepare", "--root", str(tmp_path / "heldout-root"), "--output", str(heldout),
           "--synthetic-heldout", "--per-class", "3", "--seed", "7"], capsys)
    corruption_paths = []
    for name, corruption in (("synthetic_blur_s1", "gaussian_blur"), ("synthetic_saturate_s1", "saturate")):
        destination = tmp_path / name
        _call(["prepare", "--root", str(tmp_path / "heldout-root"), "--output", str(destination),
               "--synthetic-corruption-from", str(heldout), "--corruption", corruption,
               "--severity", "1", "--seed", "23"], capsys)
        corruption_metadata = json.loads((destination / "metadata.json").read_text())["metadata"]
        assert corruption_metadata["corruption"] == corruption
        assert corruption_metadata["severity"] == 1 and corruption_metadata["synthetic"] is True
        corruption_paths.append(f"{name}={destination}")
    outputs = []
    for run in runs["runs"]:
        result = _call(["evaluate", "--root", str(study), "--run-id", run["run_id"],
                        "--synthetic-heldout", str(heldout), "--indices", "0", "1",
                        "--synthetic-corruption", corruption_paths[0],
                        "--synthetic-corruption", corruption_paths[1]], capsys)
        outputs.append(result)
    assert len(outputs) == 6
    assert all(item["source"] == "synthetic-heldout" and item["evidence_eligible"] is False for item in outputs)
    assert all(set(item["conditions"]) == {"clean", *[entry.split("=", 1)[0] for entry in corruption_paths]}
               for item in outputs)

    analysis_dir = tmp_path / "analysis"
    summary = _call(["analyze", "--run-table", str(study), "--output", str(analysis_dir),
                     "--include-synthetic", "--cohort", "smoke"], capsys)
    expected_kinds = {"accuracy_vs_inference_k", "accuracy_vs_inference_macs", "accuracy_vs_fraction",
                      "accuracy_vs_severity_by_k", "state_norms_and_changes_vs_k", "ef_ablation_effects"}
    assert {item["kind"] for item in summary["plot_data"]} == expected_kinds
    assert all((analysis_dir / name).is_file() and (analysis_dir / name).stat().st_size > 0
               for name in summary["plots"])
    assert (analysis_dir / "summary.json").is_file()
    assert (study / "preregistration.json").is_file()
    assert (study / "cohorts" / "smoke" / "freeze.json").is_file()
    assert all((study / "results" / "smoke" / run["run_id"] / "test_metrics.json").is_file()
               for run in runs["runs"])

    profile_path = tmp_path / "profile.json"
    profile = _call(["profile", "--family", "E", "--depth", "8", "--width", "1",
                     "--device", "cpu", "--warmup", "1", "--repeats", "1", "--batch-size", "1",
                     "--inference-grid", "1", "2", "8", "--sweep-examples", "1",
                     "--output", str(profile_path)], capsys)
    assert profile["training_path_distribution"]["finite_updates"] >= 2
    assert "training_objective" in profile["timing_scope"]["training_step"]
    assert profile_path.is_file()
    freeze_path = study / "cohorts" / "smoke" / "freeze.json"
    freeze_bytes = freeze_path.read_bytes()
    tampered = json.loads(freeze_bytes)
    tampered["seal"] = "0" * 64
    freeze_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert cli.main(["evaluate", "--root", str(study), "--run-id", runs["runs"][0]["run_id"],
                     "--synthetic-heldout", str(heldout), "--indices", "0", "1",
                     "--synthetic-corruption", corruption_paths[0],
                     "--synthetic-corruption", corruption_paths[1]]) == 1
    capsys.readouterr()
    freeze_path.write_bytes(freeze_bytes)


def test_matrix_and_evaluation_guards_precede_training_or_official_loaders(tmp_path, capsys, monkeypatch):
    training = tmp_path / "training"
    _call(["prepare", "--root", str(tmp_path / "data-root"), "--output", str(training),
           "--synthetic", "--per-class", "3", "--split-seed", "19"], capsys)
    bad_config = _write_config(tmp_path / "wrong-population.yaml", split_seed=1729)
    train_calls = []
    monkeypatch.setattr(cli.engine, "train_run", lambda *a, **k: train_calls.append((a, k)))
    assert cli.main(["smoke", "--config", str(bad_config), "--data", str(training),
                     "--run-dir", str(tmp_path / "wrong-pop-run")]) == 1
    capsys.readouterr()
    assert train_calls == []
    study = tmp_path / "unpreregistered"
    _call(["matrix", "plan", "--root", str(study), "--data", str(training), "--amendment"], capsys)
    trained = []
    monkeypatch.setattr(protocol, "train_run", lambda *a, **k: trained.append((a, k)))
    assert cli.main(["matrix", "execute", "--root", str(study), "--data", str(training), "--execute"]) == 1
    capsys.readouterr()
    assert trained == []

    heldout = tmp_path / "heldout"
    _call(["prepare", "--root", str(tmp_path / "heldout-root"), "--output", str(heldout),
           "--synthetic-heldout", "--per-class", "3", "--seed", "4"], capsys)
    invalid_corruption = tmp_path / "invalid-corruption"
    assert cli.main(["prepare", "--root", str(tmp_path / "heldout-root"), "--output",
                     str(invalid_corruption), "--synthetic-corruption-from", str(heldout),
                     "--corruption", "gaussian_blur", "--severity", "6", "--seed", "9"]) == 1
    capsys.readouterr()
    assert not invalid_corruption.exists()
    corruption_args = []
    for name, mode in (("synthetic_blur_s1", "gaussian_blur"), ("synthetic_spatter_s1", "spatter")):
        destination = tmp_path / name
        _call(["prepare", "--root", str(tmp_path / "heldout-root"), "--output", str(destination),
               "--synthetic-corruption-from", str(heldout), "--corruption", mode,
               "--severity", "1", "--seed", "9"], capsys)
        corruption_args.extend(["--synthetic-corruption", f"{name}={destination}"])
    loader_calls = []
    monkeypatch.setattr(protocol, "_open_official_test", lambda *a, **k: loader_calls.append("test"))
    monkeypatch.setattr(protocol, "_open_cifar100_c", lambda *a, **k: loader_calls.append("c"))
    assert cli.main(["evaluate", "--root", str(study), "--run-id", "unknown",
                     "--synthetic-heldout", str(training), *corruption_args]) == 1
    capsys.readouterr()
    assert cli.main(["evaluate", "--root", str(study), "--run-id", "unknown",
                     "--official-data-root", str(tmp_path / "official-test"),
                     "--cifar100c-root", str(tmp_path / "official-c")]) == 1
    capsys.readouterr()
    assert cli.main(["evaluate", "--root", str(study), "--run-id", "unknown",
                     "--synthetic-heldout", str(heldout), *corruption_args]) == 1
    capsys.readouterr()
    assert trained == [] and loader_calls == []
