# H8 hybrid follow-up

This is a separate **exploratory, post-original-test follow-up**. The original B8/C8/E8 test outcomes were already known when H8 was proposed. Its protocol is prospective for the new H runs only; it does not turn the existing test partition into an unseen confirmatory dataset.

H8 executes `[A,S,S,S,S,S,S,Z]`: independent endpoint blocks and six applications of one shared central block. There are three unique residual blocks, width 32, eight applications, **59,844 trainable parameters**, and **37,973,120 forward MACs/example** under the original counting convention. H8 is larger than C8, so a difference cannot isolate endpoint specialization from additional capacity. Parameter count is not a latency measurement.

The frozen historical B8/C8 controls are reused, not retrained. Their freeze SHA-256 remains:

```text
64da1d811fb71a1913042d50efc62fdda49d240b70e2f54c375ebbe328810bd0
```

## Fixed recipe and planned comparisons

[`protocol.json`](protocol.json) is authoritative. Seeds are **11, 23, 37, 53, 71**, each with 10,000 updates, batch 128, 4,500 training examples, validation size 5,000, split seed 1729, SGD/momentum 0.9/weight decay 0.0005, learning rate 0.03, cosine horizon 10,000, and no AMP. Selection uses validation K8 accuracy, then cross-entropy, then earliest update, at a 200-update cadence. Recall, variable depth, progressive objectives, K32, CIFAR-100-C, additional seeds, and control retraining are outside this protocol.

The two planned accuracy contrasts are **H8−B8** and **H8−C8**. Report arithmetic means, sample standard deviations, five paired seed-level deltas, and two-sided nominal Student-t 95% intervals with four degrees of freedom. Both comparisons are exploratory and unadjusted for multiplicity. No post-H-test tuning is authorized.

## Isolated operator lifecycle

`run_hybrid.py` never downloads data. A caller must prepare a **new, exclusively owned root**, prove its resource boundaries, and arrange train/meta input copies before starting. Never point it at a user, shared, or historical experiment directory.

Required layout:

```text
<ROOT>/
  source/                         # immutable source snapshot
    src/recurrent_cnn/
    benchmarks/hybrid-generalization-20261015/
    benchmarks/minimal-generalization-20261015/
    benchmarks/duration-calibration-20261015/
  source_hashes.json               # {"files": {"relative/POSIX/path": "sha256", ...}}
  protocol.json                   # byte-identical to the source snapshot copy
  resource-proof.json
  data/cifar-100-python/train
  data/cifar-100-python/meta
```

The source manifest must cover the complete native package, H operator/protocol, reused historical helper, calibration helper, and all frozen control inputs. Those inputs include the original protocol, preregistration, freeze, validation/test metric JSON, and the ten compressed B8/C8 update logs. Relative manifest keys use `/`, including on Windows. The analysis source can be frozen in the same snapshot.

The resource proof must establish `classification: DISPOSABLE_VERIFIED`, fresh root ownership, protected input/interpreter boundaries, private runtime/cache locations, no destructive hooks, `cpu_threads` within 80% of the host's logical CPUs, and `cuda_memory_fraction: 0.5` for CUDA work. It is an inventory receipt, not a way to manufacture isolation by assertion. No symlink, hardlink, reparse-point, shared database, remote service, or fallback may escape the declared boundary.

Set `PYTHONDONTWRITEBYTECODE=1` and redirect all required environment paths inside the run root before import:

```text
HOME USERPROFILE TMP TEMP TMPDIR APPDATA LOCALAPPDATA XDG_CACHE_HOME
MPLCONFIGDIR TORCH_HOME CUDA_CACHE_PATH TORCHINDUCTOR_CACHE_DIR
TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR
```

Also keep configuration caches private. The workload guard rejects external writes, sockets/downloads, premature official-test access, and prohibited workloads. It does not reserve hardware: the coordinator must confirm the exact target GPU is free of other ML jobs before each CUDA launch and preserve desktop applications. On Windows WDDM, existing desktop processes can appear as `C+G`; a nonempty NVIDIA process list alone does not establish a competing training job.

With the root, environment, immutable snapshot, and its full 40-character Git source SHA prepared, use this order:

```text
python <ROOT>/source/benchmarks/hybrid-generalization-20261015/run_hybrid.py --root <ROOT> --source-head <SHA> --phase probe --device cuda
python <ROOT>/source/benchmarks/hybrid-generalization-20261015/run_hybrid.py --root <ROOT> --source-head <SHA> --phase train
python <ROOT>/source/benchmarks/hybrid-generalization-20261015/run_hybrid.py --root <ROOT> --source-head <SHA> --phase verify
```

The probe uses only synthetic data for two updates; it is not scientific evidence about accuracy. CPU probe is also supported. Prove first progress and set a no-progress deadline before scaling the launcher. Close child stdin and retain failures; do not silently retry or overwrite evidence.

Training refuses an existing study directory or a staged official-test file. It first binds the recipe, data identities, and init/minibatch/augmentation seeds to the historical controls. After each run, every update's sample and augmentation digest must match both paired controls. It requires five complete finite 10,000-update logs, independently recomputes validation selection, seals selected checkpoints, freezes their hashes/source/protocol, and evaluates validation only.

**Do not stage the official-test partition until the five-run freeze has passed independent read-only verification.** Verify checkpoint/log/run hashes, recipe, sample/augmentation pairing, cadence, selector, and source/protocol identity. Input-identity hashing by the coordinator is not H model evaluation; workload access counters cover the guarded operator, not all historical or coordinator reads.

Only then stage the pinned official-test bytes as a new run-owned copy and invoke `--phase test`. It re-verifies the freeze before opening the test gate and creates an exclusive evaluation marker. Repeated evaluation and training after test opening are not supported. A partial failure is evidence to preserve and investigate, not permission to resume, reset, or rerun a different protocol.

## Analysis and regeneration

`analyze_hybrid.py` validates H checkpoint, recipe, split/subset/data identities and frozen B8/C8 control pairing. `scripts/make_hybrid_figures.py` renders three deterministic SVGs, machine-readable figure data, a table, and the English report. Use fresh output directories; see the generated report's commands after real results are available. Model weights and CIFAR image partitions are not publication artifacts.

The focused tests cover actual-width H8 gradient accumulation and checkpoint resumption, bit-exact preservation of B8/C8 against the original source, costs, lifecycle gates, malformed evidence, paired statistics, and deterministic rendering. None of these synthetic checks substitutes for the five scientific runs and separately gated official evaluation.
