#!/usr/bin/env python3
"""Isolated operator for the preregistered END/INNER position ablation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PROTOCOL_REL = Path("benchmarks/position-ablation-20261016/protocol.json")
SEEDS = (11, 23, 37, 53, 71)
ARMS = ("END", "INNER")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def event(name, started, **fields):
    print(json.dumps({"event": name, "elapsed_seconds": time.perf_counter()-started,
                      "utc": datetime.now(timezone.utc).isoformat(), **fields}, sort_keys=True, allow_nan=False), flush=True)


def _helper(source):
    path = source / "benchmarks/minimal-generalization-20261015/run_study.py"
    spec = importlib.util.spec_from_file_location("position_native_study_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native study guards")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hashes(source):
    calibration = source / "benchmarks/duration-calibration-20261015/run_calibration.py"
    spec = importlib.util.spec_from_file_location("position_source_hash_helper", calibration)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native source-manifest verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_source_hashes(source)


def _root(root_arg, source_head):
    root = root_arg.resolve(strict=True)
    if not root.is_dir() or len(source_head) != 40 or any(c not in "0123456789abcdef" for c in source_head):
        raise ValueError("--root must exist and --source-head must be a full lowercase SHA")
    proof_path = root / "resource-proof.json"
    if not proof_path.resolve().is_relative_to(root):
        raise PermissionError("resource proof must be contained in --root")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("classification") != "DISPOSABLE_VERIFIED" or Path(proof.get("filesystem_root", "")).resolve() != root:
        raise PermissionError("resource proof must certify this exact filesystem root as DISPOSABLE_VERIFIED")
    source = (root / "source").resolve(strict=True)
    if not source.is_relative_to(root) or not (source / "src/recurrent_cnn").is_dir():
        raise PermissionError("source snapshot must be contained at <root>/source")
    identity_path=root/"source-identity.json"
    if not identity_path.resolve().is_relative_to(root) or not (root/"source_hashes.json").resolve().is_relative_to(root):
        raise PermissionError("source identity/manifest must be contained in --root")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("source_head") != source_head:
        raise ValueError("source-identity.json does not match --source-head")
    hashes = _hashes(source)
    protocol_path = root / "protocol.json"
    snapshot_protocol = source / PROTOCOL_REL
    if not protocol_path.resolve().is_relative_to(root) or not snapshot_protocol.resolve().is_relative_to(source):
        raise PermissionError("protocol copies must remain inside their staged roots")
    if protocol_path.read_bytes() != snapshot_protocol.read_bytes():
        raise ValueError("staged protocol differs from immutable source snapshot")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "position-ablation-protocol-v1" or protocol.get("experiment_id") != "position-ablation-20261016":
        raise ValueError("unexpected position-ablation protocol identity")
    expected_model={"native_base_family":"H","depth":8,"width":32,"alpha":0.1,"num_classes":100,
        "unique_branches":3,"branch_use_counts":[1,6,1],"parameters":59844,"forward_macs_per_example":37973120}
    expected_recipe={"paired_seeds":list(SEEDS),"run_order":"seed-major, END then INNER","split_seed":1729,
        "fraction":0.1,"train_examples_per_seed":4500,"validation_examples":5000,"batch_size":128,
        "total_updates_per_run":10000,"expected_scientific_runs":10,"expected_scientific_updates":100000,
        "cosine_horizon":10000,"validation_cadence":200,"optimizer":{"name":"SGD","learning_rate":0.03,"momentum":0.9,"weight_decay":0.0005},
        "device":"cuda","amp":False,"selection_identity":"amended_v2","inference_grid":[8]}
    if any(protocol["model"].get(k)!=v for k,v in expected_model.items()):
        raise ValueError("protocol model does not match the fixed position-ablation contract")
    if any(protocol["recipe"].get(k)!=v for k,v in expected_recipe.items()):
        raise ValueError("protocol recipe does not match the fixed position-ablation contract")
    if {k:tuple(v["branch_map"]) for k,v in protocol["arms"].items()}!={"END":(0,1,1,1,1,1,1,2),"INNER":(1,0,1,1,1,1,2,1)}:
        raise ValueError("protocol branch maps differ from the fixed treatments")
    for key in ("HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "APPDATA", "LOCALAPPDATA", "XDG_CACHE_HOME",
                "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
        if not os.environ.get(key) or not Path(os.environ[key]).resolve().is_relative_to(root):
            raise PermissionError(f"{key} must resolve inside --root")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise PermissionError("PYTHONDONTWRITEBYTECODE=1 is required")
    return root, source, proof, hashes, protocol, sha256_file(protocol_path), sha256_file(proof_path)


def _native(source, helper):
    sys.path.insert(0, str((source / "src").resolve()))
    np, torch, data, engine, models, TrainConfig = helper._native_import(source)
    for module in (data, engine, models):
        if not Path(module.__file__).resolve().is_relative_to((source / "src").resolve()):
            raise RuntimeError(f"unexpected module import origin: {module.__name__}")
    position_path = source / PROTOCOL_REL.parent / "position_model.py"
    spec = importlib.util.spec_from_file_location("position_mapped_model", position_path)
    position = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(position)
    return np, torch, data, engine, models, TrainConfig, position


def _config(TrainConfig, seed, data_hash="", split_hash="", subset_hash="", probe=False):
    return TrainConfig.from_mapping(dict(family="H", depth=8, width=32, alpha=0.1, fraction=0.1,
        paired_seed=seed, data_hash=data_hash, split_hash=split_hash, subset_hash=subset_hash,
        role="research", synthetic=probe, total_updates=2 if probe else 10000, batch_size=128,
        learning_rate=0.03, momentum=0.9, weight_decay=0.0005, cosine_horizon=2 if probe else 10000,
        validation_cadence=1 if probe else 200, device="cpu" if probe else "cuda", amp=False,
        num_classes=100, split_seed=1729, selection_identity="amended_v2"))


def _manifest_arrays(data, np, arrays, seed):
    population, validation = data.make_split(arrays.labels, 1729)
    subset = data.make_subsets(arrays.labels, population, seed)[0.1]
    if len(subset) != 4500 or len(validation) != 5000 or not np.array_equal(
            np.bincount(arrays.labels[subset], minlength=100), np.full(100, 45)):
        raise ValueError("native split/subset dimensions or class balance mismatch")
    return population, subset, validation


def _identity(arm, seed, branch_map, protocol_hash, cfg, data_hash, split_hash, subset_hash):
    return {"schema": "position-ablation-treatment-v1", "arm": arm, "paired_seed": seed,
            "branch_map": list(branch_map), "protocol_sha256": protocol_hash,
            "config": cfg.to_dict(), "config_hash": cfg.config_hash,
            "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash}


def _assert_clean(helper):
    helper._assert_clean_counters()


def _model_receipt(torch, models, position, costs, x, seed=31):
    spec = models.ModelSpec("H", 8, width=32, alpha=0.1, num_classes=100)
    c = models.build_model(models.ModelSpec("C", 8, width=32, alpha=0.1, num_classes=100), seed=seed)
    end = position.build_position_model(spec, seed=seed, arm="END")
    inner = position.build_position_model(spec, seed=seed, arm="INNER")
    for model in (end, inner):
        for name in ("stem", "head"):
            assert all(torch.equal(a, b) for a, b in zip(getattr(model, name).parameters(), getattr(c, name).parameters()))
        assert len({id(p) for p in model.parameters()}) == len(list(model.parameters()))
    assert len({id(p) for m in (end,) for p in m.branches.parameters()}) == len(list(end.branches.parameters()))
    for index in range(3):
        assert end.branches[index] is not end.branches[(index+1) % 3]
        assert all(torch.equal(a, b) for a, b in zip(end.branches[0].parameters(), end.branches[index].parameters()))
    x = x.cpu()
    with torch.no_grad():
        c_logits, end_logits, inner_logits = c(x), end(x), inner(x)
    if not torch.equal(c_logits, end_logits) or not torch.equal(c_logits, inner_logits):
        raise AssertionError("initial C, END and INNER logits differ")
    counts = [0, 0, 0]
    for model in (end, inner):
        counts[:] = [0, 0, 0]
        hs = [b.register_forward_hook(lambda _m, _i, _o, j=j: counts.__setitem__(j, counts[j]+1)) for j,b in enumerate(model.branches)]
        model(x)
        for h in hs: h.remove()
        if counts != [1, 6, 1]:
            raise AssertionError(f"invalid mapped branch usage: {counts}")
    params = costs.parameter_count(end)
    macs = costs.mac_breakdown(spec, (1,3,32,32), inference_k=8)["mac_breakdown"]["total"]
    if (params, macs) != (59844, 37973120):
        raise ValueError(f"cost mismatch: {params} parameters, {macs} MACs")
    state_hashes={}
    for name,model in (("C",c),("END",end),("INNER",inner)):
        digest=hashlib.sha256()
        for key,value in sorted(model.state_dict().items()):
            digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        state_hashes[name]=digest.hexdigest()
    return {"initial_logits_sha256": hashlib.sha256(c_logits.numpy().tobytes()).hexdigest(),
            "initial_state_sha256":state_hashes,"parameters": params, "forward_macs": macs, "branch_use_counts": [1,6,1]}


def _probe(root, source_head, source, helper, source_hashes, protocol_hash, proof, np, torch, data, engine, models, TrainConfig, position, device):
    probe = root / "probe"
    probe.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    arrays = data.prepare_data(probe / "unused", download=False, synthetic=True, synthetic_per_class=3, seed=0)
    train_all, val = data.make_split(arrays.labels, 1729, synthetic=True)
    train = data.make_subsets(arrays.labels, train_all, 11, synthetic=True)[0.1]
    dh = engine._array_identity(arrays); sh, suh = engine.training_identity_hashes(len(arrays.labels), train, val)
    cfg = _config(TrainConfig, 11, dh, sh, suh, probe=True)
    x = torch.randn(2,3,32,32)
    model_receipt = _model_receipt(torch, models, position, __import__("recurrent_cnn.costs", fromlist=["costs"]), x)
    started = time.perf_counter()
    for arm in ARMS:
        cfg_arm = cfg
        run = probe / "runs" / f"{arm}-s11"
        with position.mapped_engine_factory(engine, arm):
            result = engine.train_run(cfg_arm, arrays, train, val, run)
        if result["status"] != "complete" or result["update"] != 2:
            raise RuntimeError("mapped synthetic probe did not complete two updates")
        rows = helper._records(run / "metrics.jsonl")
        if len(rows) != 2 or any(r.get("finite") is not True for r in rows):
            raise ValueError("probe produced invalid native training metrics")
        payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        with position.mapped_engine_factory(engine, arm):
            sweep = engine.evaluate_checkpoint_sweep(payload, arrays, val, device=device, inference_grid=[8], batch_size=256)
        expected = position.BRANCH_MAPS[arm]
        rebuilt = position.build_position_model(models.ModelSpec(**payload["model_spec"]), seed=0, arm=arm)
        rebuilt.load_state_dict(payload["model_state"])
        if rebuilt.branch_map != expected or sweep["evaluated_grid"] != [8]:
            raise ValueError("mapped checkpoint reconstruction or sweep failed")
    uninterrupted = probe / "resume-full"
    split = probe / "resume-split"
    tiny = _config(TrainConfig, 11, dh, sh, suh, probe=True)
    engine.train_run(tiny, arrays, train, val, uninterrupted)
    engine.train_run(tiny, arrays, train, val, split, step_limit=1)
    engine.train_run(tiny, arrays, train, val, split, resume=True)
    if (uninterrupted / "metrics.jsonl").read_bytes() != (split / "metrics.jsonl").read_bytes():
        raise AssertionError("native resume differs from uninterrupted training")
    original = engine.build_model
    try:
        try:
            with position.mapped_engine_factory(engine, "END"):
                raise RuntimeError("context restoration sentinel")
        except RuntimeError:
            pass
        if engine.build_model is not original: raise AssertionError("engine factory was not restored after exception")
    finally:
        engine.build_model = original
    _assert_clean(helper)
    receipt = {"schema":"position-ablation-probe-v1", "scientific":False, "source_head":source_head,
        "protocol_sha256":protocol_hash, "models":model_receipt, "updates_per_arm":2,
        "runtime":{"device":device,"cpu_threads":torch.get_num_threads()}, "source_hashes":source_hashes,
        "counters":helper.COUNTERS, "duration_seconds":time.perf_counter()-started}
    helper._json_exclusive(probe / "probe_receipt.json", receipt)
    event("probe_complete", started, arms=2, updates=4)


def _record_stream(rows):
    return hashlib.sha256(json.dumps([(r["sample_digest"],r["augmentation_digest"]) for r in rows], separators=(",", ":")).encode()).hexdigest()


def _train(root, source_head, helper, source_hashes, protocol, protocol_hash, proof_hash, np, torch, data, engine, models, TrainConfig, position):
    study = root / "study"
    if study.exists() or (root / "data/cifar-100-python/test").exists():
        raise FileExistsError("scientific train refuses an existing study or official-test file")
    data_root = root / "data"
    train_file, meta_file = data_root / "cifar-100-python/train", data_root / "cifar-100-python/meta"
    if not data_root.resolve().is_relative_to(root) or not train_file.resolve().is_relative_to(root) or not meta_file.resolve().is_relative_to(root):
        raise PermissionError("data and staged CIFAR files must resolve inside --root")
    if not train_file.is_file() or not meta_file.is_file() or (data_root / "cifar-100-python/test").exists():
        raise PermissionError("train requires only owned CIFAR training and metadata files; test must be absent")
    arrays = data.prepare_data(data_root, download=False, synthetic=False)
    if arrays.namespace != "train" or len(arrays.labels) != 50000 or not np.array_equal(np.bincount(arrays.labels, minlength=100), np.full(100,500)):
        raise ValueError("expected exactly 50,000 native training examples, 500 per class")
    data_hash = engine._array_identity(arrays)
    splits, configs, identities = {}, {}, {}
    for seed in SEEDS:
        population, subset, val = _manifest_arrays(data,np,arrays,seed)
        sh,suh = engine.training_identity_hashes(len(arrays.labels),subset,val)
        cfg = _config(TrainConfig,seed,data_hash,sh,suh)
        splits[seed]=(subset,val,sh,suh)
        for arm in ARMS:
            rid=f"{arm}-s{seed}"; configs[rid]=cfg.to_dict()
            identities[rid]=_identity(arm,seed,position.BRANCH_MAPS[arm],protocol_hash,cfg,data_hash,sh,suh)
    study.mkdir(exist_ok=False)
    prereg={"schema":"position-ablation-preregistration-v1","source_head":source_head,"source_hashes":source_hashes,
        "protocol_sha256":protocol_hash,"resource_proof_sha256":proof_hash,"data_hash":data_hash,
        "configs":configs,"treatments":identities,"roster":[f"{a}-s{s}" for s in SEEDS for a in ARMS],
        "initial_state_receipt":_model_receipt(torch,models,position,__import__("recurrent_cnn.costs",fromlist=["costs"]),torch.randn(2,3,32,32)),
        "created_utc":datetime.now(timezone.utc).isoformat()}
    helper._json_exclusive(study/"preregistration.json",prereg)
    prereg_hash=sha256_file(study/"preregistration.json")
    for seed in SEEDS:
        subset,val,sh,suh=splits[seed]
        for arm in ARMS:
            rid=f"{arm}-s{seed}"; run=study/"runs"/rid
            cfg=TrainConfig.from_mapping(configs[rid]); event("training_start",time.perf_counter(),source_head=source_head,arm=arm,seed=seed,update=0)
            original=engine._append_record; last=[0.0]
            def progress(path,record,_append=original,_arm=arm,_seed=seed):
                result=_append(path,record)
                if record["update"]%200==0: event("training_progress",last[0],source_head=source_head,arm=_arm,seed=_seed,update=record["update"])
                return result
            engine._append_record=progress; last[0]=time.perf_counter()
            try:
                with position.mapped_engine_factory(engine,arm): result=engine.train_run(cfg,arrays,subset,val,run)
            finally: engine._append_record=original
            rows=helper._records(run/"metrics.jsonl")
            checks=list(range(200,10001,200))
            if result.get("status")!="complete" or result.get("update")!=10000 or len(rows)!=10000 or [r["update"] for r in rows if r.get("validation_accuracy") is not None]!=checks:
                raise RuntimeError(f"native training run incomplete or cadence mismatch: {rid}")
            if any(r.get("finite") is not True for r in rows): raise ValueError(f"nonfinite update in {rid}")
            if sum(int(r["examples"]) for r in rows if r["update"]==1)!=128: raise ValueError("unexpected sample accounting")
            if len({r["sample_digest"] for r in rows})==0: raise ValueError("missing sample stream")
            pick=max((r for r in rows if r.get("validation_accuracy") is not None),key=lambda r:(r["validation_accuracy"],-r["validation_loss"],-r["update"]))
            payload=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
            if payload.get("update")!=pick["update"] or payload.get("selection",{}).get("identity")!="amended_v2": raise ValueError("native selected checkpoint differs from preregistered selector")
            treatment=identities[rid]
            sealed=study/"sealed"/rid; sealed.mkdir(parents=True,exist_ok=False)
            shutil.copyfile(run/"best.pt",sealed/"selected.pt"); helper._json_exclusive(sealed/"treatment.json",treatment)
            event("training_complete",last[0],source_head=source_head,arm=arm,seed=seed,update=10000)
    for seed in SEEDS:
        end_rows=helper._records(study/"runs"/f"END-s{seed}"/"metrics.jsonl")
        inner_rows=helper._records(study/"runs"/f"INNER-s{seed}"/"metrics.jsonl")
        if [(r.get("sample_digest"),r.get("augmentation_digest")) for r in end_rows] != [(r.get("sample_digest"),r.get("augmentation_digest")) for r in inner_rows]:
            raise ValueError(f"paired sample/augmentation stream differs for seed {seed}")
    freeze_runs=[]
    for seed in SEEDS:
        subset,val,sh,suh=splits[seed]
        for arm in ARMS:
            rid=f"{arm}-s{seed}"; run=study/"runs"/rid; rows=helper._records(run/"metrics.jsonl")
            pick=max((r for r in rows if r.get("validation_accuracy") is not None),key=lambda r:(r["validation_accuracy"],-r["validation_loss"],-r["update"]))
            frozen=study/"sealed"/rid/"selected.pt"
            freeze_runs.append({"run_id":rid,"arm":arm,"paired_seed":seed,"branch_map":list(position.BRANCH_MAPS[arm]),
                "config":configs[rid],"data_hash":data_hash,"split_hash":sh,"subset_hash":suh,
                "selected_update":pick["update"],"selected_train_accuracy":pick["clean_train_accuracy"],
                "selected_validation_accuracy":pick["validation_accuracy"],"selected_validation_cross_entropy":pick["validation_loss"],
                "run_sha256":sha256_file(run/"run.json"),"log_sha256":sha256_file(run/"metrics.jsonl"),
                "checkpoint_sha256":sha256_file(frozen),"treatment_sha256":sha256_file(study/"sealed"/rid/"treatment.json"),
                "paired_stream_sha256":_record_stream(rows)})
    freeze={"schema":"position-ablation-freeze-v1","source_head":source_head,"source_hashes":source_hashes,
        "protocol_sha256":protocol_hash,"preregistration_sha256":prereg_hash,"runs":freeze_runs,
        "created_utc":datetime.now(timezone.utc).isoformat()}
    freeze["canonical_sha256"]=canonical(freeze); helper._json_exclusive(study/"freeze.json",freeze)
    freeze_hash=sha256_file(study/"freeze.json"); rows_out=[]
    validation_arrays=data.CifarArrays(arrays.images,arrays.labels,arrays.namespace,{**(arrays.metadata or {}),"source":"validation","dataset_id":"cifar100"})
    for seed in SEEDS:
        subset,val,sh,suh=splits[seed]
        for arm in ARMS:
            rid=f"{arm}-s{seed}"; run=study/"runs"/rid; path=study/"sealed"/rid/"selected.pt"
            with position.mapped_engine_factory(engine,arm):
                sweep=engine.evaluate_checkpoint_sweep(path,validation_arrays,val,device="cuda",inference_grid=[8],batch_size=256)
            metric=sweep["metrics"][8] if 8 in sweep.get("metrics",{}) else sweep["metrics"]["8"]
            logs=helper._records(run/"metrics.jsonl"); pick=max((r for r in logs if r.get("validation_accuracy") is not None),key=lambda r:(r["validation_accuracy"],-r["validation_loss"],-r["update"]))
            if abs(pick["validation_accuracy"]-metric["top1"])>1e-12 or abs(pick["validation_loss"]-metric["cross_entropy"])>2e-5:
                raise ValueError(f"native selected metrics and mapped checkpoint sweep disagree: {rid}")
            gap=pick["clean_train_accuracy"]-metric["top1"]
            fixed=logs[-1]
            rows_out.append({"arm":arm,"paired_seed":seed,"branch_map":list(position.BRANCH_MAPS[arm]),"selected_update":pick["update"],
                "checkpoint_sha256":sha256_file(path),"data_hash":data_hash,"split_hash":sh,"subset_hash":suh,
                "metrics":{k:metric[k] for k in ("top1","cross_entropy","ece","ece_bins")},
                "selected_train_accuracy":pick["clean_train_accuracy"],"selected_gap":gap,
                "fixed_budget":{"update":10000,"train_accuracy":fixed["clean_train_accuracy"],"validation_accuracy":fixed["validation_accuracy"],
                    "validation_cross_entropy":fixed["validation_loss"],"gap":fixed["clean_train_accuracy"]-fixed["validation_accuracy"]},
                "parameters":59844,"forward_macs":37973120,"training_updates":10000,"paired_stream_sha256":_record_stream(logs)})
    validation_doc={"schema":"position-ablation-validation-v1","split":"validation","source_head":source_head,
        "protocol_sha256":protocol_hash,"freeze_sha256":freeze_hash,"rows":rows_out}
    helper._json_exclusive(study/"validation.json",validation_doc)
    _assert_clean(helper)
    helper._json_exclusive(study/"train_receipt.json",{"status":"complete","source_head":source_head,"source_hashes":source_hashes,
        "protocol_sha256":protocol_hash,"freeze_sha256":freeze_hash,"runs":10,"updates":100000,"counters":helper.COUNTERS})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--source-head",required=True); parser.add_argument("--phase",required=True,choices=("probe","train","verify"))
    parser.add_argument("--device",choices=("cpu","cuda"),default=None); args=parser.parse_args(argv)
    if args.device is not None and args.phase!="probe": parser.error("--device is permitted only for probe")
    root,source,proof,source_hashes,protocol,protocol_hash,proof_hash=_root(args.root,args.source_head)
    study=root/"study"
    if args.phase=="probe" and (root/"probe").exists(): raise FileExistsError("probe directory already exists")
    if args.phase=="train" and (study.exists() or (root/"data/cifar-100-python/test").exists()): raise FileExistsError("train refuses existing study or official-test data")
    if args.phase=="verify" and not study.is_dir(): raise FileNotFoundError("study does not exist")
    # The borrowed native guard recognizes the established private cache roots; alias
    # the additional config/pip locations to those already-contained roots.
    os.environ["XDG_CONFIG_HOME"]=os.environ["XDG_CACHE_HOME"]
    os.environ["PIP_CACHE_DIR"]=os.environ["TORCH_HOME"]
    helper=_helper(source); helper.install_guards(root); started=time.perf_counter()
    try:
        np,torch,data,engine,models,TrainConfig,position=_native(source,helper)
        if args.phase=="verify":
            torch.set_num_threads(1)
        original_test=data._open_official_test
        def deny_test(*a,**kw): raise PermissionError("official-test partition is permanently prohibited")
        data._open_official_test=deny_test
        if args.phase=="probe":
            device=args.device or "cpu"
            if device!="cpu": raise PermissionError("position probe is synthetic CPU-only")
            _probe(root,args.source_head,source,helper,source_hashes,protocol_hash,proof,np,torch,data,engine,models,TrainConfig,position,device)
        elif args.phase=="train":
            threads=proof.get("cpu_threads"); ncpu=os.cpu_count()
            if type(threads) is not int or ncpu is None or not 1<=threads<=math.floor(ncpu*.8): raise PermissionError("resource proof cpu_threads exceeds 80% CPU budget")
            if proof.get("cuda_memory_fraction")!=0.5 or not torch.cuda.is_available(): raise PermissionError("train requires proven 50% CUDA memory and available CUDA")
            torch.set_num_threads(threads); torch.cuda.set_per_process_memory_fraction(.5,0)
            _train(root,args.source_head,helper,source_hashes,protocol,protocol_hash,proof_hash,np,torch,data,engine,models,TrainConfig,position)
        else:
            freeze_path=study/"freeze.json"
            prereg_path=study/"preregistration.json"
            validation_path=study/"validation.json"
            if not all(p.is_file() for p in (freeze_path,prereg_path,validation_path)):
                raise FileNotFoundError("verify requires preregistration, freeze and validation receipts")
            freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
            if freeze.get("schema")!="position-ablation-freeze-v1" or freeze.get("source_head")!=args.source_head or freeze.get("protocol_sha256")!=protocol_hash:
                raise ValueError("freeze source/protocol identity mismatch")
            if freeze.get("canonical_sha256")!=canonical({k:v for k,v in freeze.items() if k!="canonical_sha256"}):
                raise ValueError("freeze canonical digest mismatch")
            if len(freeze.get("runs",[]))!=10:
                raise ValueError("freeze roster is not ten runs")
            validation=json.loads(validation_path.read_text(encoding="utf-8"))
            if (validation.get("schema")!="position-ablation-validation-v1" or validation.get("split")!="validation"
                    or validation.get("source_head")!=args.source_head or validation.get("protocol_sha256")!=protocol_hash
                    or validation.get("freeze_sha256")!=sha256_file(freeze_path) or len(validation.get("rows",[]))!=10):
                raise ValueError("validation receipt schema/roster/identity mismatch")
            expected_ids=[f"{arm}-s{seed}" for seed in SEEDS for arm in ARMS]
            if [entry.get("run_id") for entry in freeze["runs"]]!=expected_ids:
                raise ValueError("freeze roster/order mismatch")
            validation_by_id={f"{row.get('arm')}-s{row.get('paired_seed')}":row for row in validation["rows"]}
            for row in freeze["runs"]:
                run_id=row["run_id"]; run_dir=study/"runs"/run_id
                checkpoint=study/"sealed"/run_id/"selected.pt"
                treatment=study/"sealed"/run_id/"treatment.json"
                log_path=run_dir/"metrics.jsonl"; run_path=run_dir/"run.json"
                if any(sha256_file(path)!=row[key] for path,key in ((checkpoint,"checkpoint_sha256"),(treatment,"treatment_sha256"),(log_path,"log_sha256"),(run_path,"run_sha256"))):
                    raise ValueError(f"frozen artifact digest mismatch: {run_id}")
                ident=json.loads(treatment.read_text(encoding="utf-8"))
                if ident.get("branch_map")!=row.get("branch_map") or ident.get("arm")!=row.get("arm") or ident.get("protocol_sha256")!=protocol_hash:
                    raise ValueError(f"treatment reconstruction identity mismatch: {run_id}")
                records=helper._records(log_path)
                if len(records)!=10000 or any(rec.get("update")!=i or rec.get("finite") is not True for i,rec in enumerate(records,1)):
                    raise ValueError(f"native training log is incomplete or nonfinite: {run_id}")
                if _record_stream(records)!=row.get("paired_stream_sha256"):
                    raise ValueError(f"paired stream digest mismatch: {run_id}")
                pick=max((rec for rec in records if rec.get("validation_accuracy") is not None),key=lambda rec:(rec["validation_accuracy"],-rec["validation_loss"],-rec["update"]))
                if pick.get("update")!=row.get("selected_update"):
                    raise ValueError(f"selected checkpoint update differs from native selector: {run_id}")
                val_row=validation_by_id.get(run_id)
                if val_row is None or val_row.get("checkpoint_sha256")!=row.get("checkpoint_sha256") or val_row.get("branch_map")!=row.get("branch_map"):
                    raise ValueError(f"validation row does not bind frozen treatment: {run_id}")
            _assert_clean(helper)
            print(json.dumps({"status":"verified","source_head":args.source_head,"freeze_sha256":sha256_file(freeze_path),"runs":10,"counters":helper.COUNTERS},sort_keys=True),flush=True)
        if _hashes(source)!=source_hashes: raise RuntimeError("immutable source changed during run")
        _assert_clean(helper)
        return 0
    finally:
        data._open_official_test=original_test
        helper.restore_guards()

if __name__=="__main__":
    raise SystemExit(main())
