"""Synthetic CPU coverage for the isolated END/INNER treatment operator."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from recurrent_cnn.config import TrainConfig
from recurrent_cnn.data import CifarArrays
from recurrent_cnn.engine import evaluate_checkpoint_sweep, train_run
from recurrent_cnn.models import ModelSpec, build_model

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks/position-ablation-20261016"


def _load_position():
    spec = importlib.util.spec_from_file_location("position_ablation_model_test", BENCH / "position_model.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic():
    rng = np.random.default_rng(21)
    arrays = CifarArrays(rng.integers(0, 256, (6,32,32,3), dtype=np.uint8), np.array([0,1,0,1,0,1]), "synthetic")
    return arrays, np.arange(4), np.array([4,5])


def _config():
    return TrainConfig("H",8,width=32,num_classes=100,total_updates=2,batch_size=2,validation_cadence=1,
                       learning_rate=.03,selection_identity="amended_v2")


def test_maps_clone_independent_parameters_initial_logits_costs_and_engine_checkpoint(monkeypatch,tmp_path):
    position = _load_position()
    torch.set_num_threads(1)
    spec = ModelSpec("H",8,width=32,num_classes=100)
    x = torch.randn(1,3,32,32)
    native_c = build_model(ModelSpec("C",8,width=32,num_classes=100),seed=9)
    models = [position.build_position_model(spec,seed=9,arm=arm) for arm in ("END","INNER")]
    for model in models:
        assert model.branch_map == position.BRANCH_MAPS[model.arm]
        params = list(model.branches.parameters())
        assert len({id(p) for p in params}) == len(params)
        assert all(torch.equal(p,q) for p,q in zip(model.branches[0].parameters(),model.branches[1].parameters()))
        assert all(torch.equal(p,q) for p,q in zip(model.branches[0].parameters(),model.branches[2].parameters()))
        with torch.no_grad():
            torch.testing.assert_close(model(x),native_c(x),rtol=0,atol=0)
        calls=[0,0,0]
        hooks=[b.register_forward_hook(lambda _m,_i,_o,j=j:calls.__setitem__(j,calls[j]+1)) for j,b in enumerate(model.branches)]
        model(x)
        for hook in hooks: hook.remove()
        assert calls == [1,6,1]

    from recurrent_cnn.costs import mac_breakdown,parameter_count
    assert parameter_count(models[0]) == 59844
    assert mac_breakdown(spec,inference_k=8)["mac_breakdown"]["total"] == 37973120
    arrays,train,val=_synthetic(); cfg=_config()
    import recurrent_cnn.engine as engine
    original=engine.build_model
    with position.mapped_engine_factory(engine,"INNER"):
        result=train_run(cfg,arrays,train,val,tmp_path/"inner")
        payload=torch.load(tmp_path/"inner/best.pt",map_location="cpu",weights_only=True)
        rebuilt=engine.build_model(ModelSpec(**payload["model_spec"]),seed=0)
        rebuilt.load_state_dict(payload["model_state"])
        sweep=evaluate_checkpoint_sweep(payload,arrays,val,inference_grid=[8])
        assert result["status"] == "complete"
        assert sweep["evaluated_grid"] == [8]
        assert rebuilt.branch_map == position.BRANCH_MAPS["INNER"]
    assert engine.build_model is original


def test_native_resume_matches_uninterrupted_and_factory_restores_after_error(tmp_path):
    position=_load_position(); arrays,train,val=_synthetic(); cfg=_config()
    import recurrent_cnn.engine as engine
    original=engine.build_model
    with position.mapped_engine_factory(engine,"END"):
        train_run(cfg,arrays,train,val,tmp_path/"full")
        train_run(cfg,arrays,train,val,tmp_path/"partial",step_limit=1)
        train_run(cfg,arrays,train,val,tmp_path/"partial",resume=True)
    assert (tmp_path/"full/metrics.jsonl").read_bytes() == (tmp_path/"partial/metrics.jsonl").read_bytes()
    assert engine.build_model is original
    with pytest.raises(RuntimeError):
        with position.mapped_engine_factory(engine,"END"):
            raise RuntimeError("restore sentinel")
    assert engine.build_model is original


def test_protocol_tampering_missing_isolation_official_test_and_source_escape_refused(tmp_path):
    runner_spec=importlib.util.spec_from_file_location("position_runner_integrity",BENCH/"run_position.py")
    runner=importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner)
    with pytest.raises((FileNotFoundError,PermissionError)):
        runner._root(tmp_path,"a"*40)
    bad=tmp_path/"bad"; bad.mkdir(); (bad/"resource-proof.json").write_text(json.dumps({"classification":"DISPOSABLE_VERIFIED","filesystem_root":str(tmp_path)}))
    with pytest.raises((ValueError,PermissionError,FileNotFoundError)):
        runner._root(bad,"a"*40)
    position=_load_position()
    with pytest.raises(ValueError,match="END or INNER"):
        position.build_position_model(ModelSpec("H",8),seed=1,arm="HISTORICAL")
    with pytest.raises(ValueError,match="native H"):
        position.build_position_model(ModelSpec("H",8,width=4),seed=1,arm="END")


def test_complete_cli_probe_is_synthetic_isolated_and_zero_guard_counters(tmp_path):
    root=tmp_path/"probe-root"; root.mkdir()
    source=root/"source"; source.mkdir()
    shutil.copytree(REPO/"src",source/"src")
    for rel in ("benchmarks/minimal-generalization-20261015/run_study.py",
                "benchmarks/duration-calibration-20261015/run_calibration.py",
                "benchmarks/position-ablation-20261016"):
        target=source/rel; target.parent.mkdir(parents=True,exist_ok=True)
        if (REPO/rel).is_dir(): shutil.copytree(REPO/rel,target)
        else: shutil.copyfile(REPO/rel,target)
    source_hashes={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    (root/"source_hashes.json").write_text(json.dumps({"files":source_hashes}))
    protocol=source/"benchmarks/position-ablation-20261016/protocol.json"
    shutil.copyfile(protocol,root/"protocol.json")
    (root/"source-identity.json").write_text(json.dumps({"source_head":"a"*40}))
    proof={"classification":"DISPOSABLE_VERIFIED","filesystem_root":str(root),"cpu_threads":1,"cuda_memory_fraction":.5}
    (root/"resource-proof.json").write_text(json.dumps(proof))
    for name in ("home","tmp","temp","tmpdir","userprofile","appdata","localappdata","xdg_cache","xdg_config","mplconfigdir","torch_home","cuda_cache_path","torchinductor_cache_dir","torch_extensions_dir","triton_cache_dir"):
        (root/name).mkdir()
    env={"PATH":"/usr/bin:/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","PYTHONHASHSEED":"0","PYTHONPATH":f"{root}:{source/'src'}:{source/'benchmarks/position-ablation-20261016'}","CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
    mapping={"HOME":"home","USERPROFILE":"userprofile","TMP":"tmp","TEMP":"temp","TMPDIR":"tmpdir","APPDATA":"appdata","LOCALAPPDATA":"localappdata","XDG_CACHE_HOME":"xdg_cache","XDG_CONFIG_HOME":"xdg_config","MPLCONFIGDIR":"mplconfigdir","TORCH_HOME":"torch_home","CUDA_CACHE_PATH":"cuda_cache_path","TORCHINDUCTOR_CACHE_DIR":"torchinductor_cache_dir","TORCH_EXTENSIONS_DIR":"torch_extensions_dir","TRITON_CACHE_DIR":"triton_cache_dir"}
    env.update({k:str(root/v) for k,v in mapping.items()})
    cmd=[sys.executable,"-B",str(source/"benchmarks/position-ablation-20261016/run_position.py"),"--root",str(root),"--source-head","a"*40,"--phase","probe","--device","cpu"]
    original_protocol=(root/"protocol.json").read_bytes()
    (root/"protocol.json").write_bytes(original_protocol+b" ")
    tampered=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True,timeout=60,check=False)
    assert tampered.returncode!=0 and not (root/"probe").exists()
    (root/"protocol.json").write_bytes(original_protocol)
    model_file=source/"src/recurrent_cnn/models.py"; original_model=model_file.read_bytes()
    model_file.write_bytes(original_model+b"\\n# tampered source\\n")
    tampered=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True,timeout=60,check=False)
    assert tampered.returncode!=0 and not (root/"probe").exists()
    model_file.write_bytes(original_model)
    result=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True,timeout=180,check=False)
    assert result.returncode==0,result.stdout+result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    runtime=next(item for item in events if item.get("event")=="runtime_configured")
    assert runtime["source_head"]=="a"*40 and type(runtime["pid"]) is int
    assert runtime["phase"]=="probe" and runtime["device"]=="cpu" and runtime["cpu_threads"]==1
    receipt=json.loads((root/"probe/probe_receipt.json").read_text())
    assert receipt["scientific"] is False and receipt["updates_per_arm"]==2
    assert receipt["models"]["parameters"]==59844 and receipt["models"]["forward_macs"]==37973120
    assert not any(receipt["counters"].values())
    assert not (root/"data/cifar-100-python/test").exists()
    sentinel=root/"data/cifar-100-python/test"; sentinel.parent.mkdir(parents=True); sentinel.write_bytes(b"synthetic refusal sentinel")
    train_cmd=[sys.executable,"-B",str(source/"benchmarks/position-ablation-20261016/run_position.py"),"--root",str(root),"--source-head","a"*40,"--phase","train"]
    refused=subprocess.run(train_cmd,cwd=root,env=env,text=True,capture_output=True,timeout=60,check=False)
    assert refused.returncode!=0 and not (root/"study").exists()
