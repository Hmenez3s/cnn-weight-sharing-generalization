# Provenance

This repository is a clean public export of a private original working repository. The scientific evidence and original source are preserved; private history, working notes and operational narratives are not included.

## Identity of the original

| Item | SHA |
|---|---|
| Private original repository head (hash only) | `3d1d45806f7a696882314a9078e57b94597cbcf0` |
| Frozen source head recorded in the preregistration | `a06961ddde53e8d7ae3a78b753e6ef708eeaa374` |
| `provenance/frozen-source.zip` (SHA-256) | `6afe0446bb3e9eacf329186c8fd2d882497b2e16f4e421dc4a6792f4add853fd` |

Only hashes of the private original are published; its location and history are not part of this repository.

## Frozen source archive

[`provenance/frozen-source.zip`](../provenance/frozen-source.zip) contains the 18 original source files below, with their original bytes. The hashes match the `original_source_sha256` map in [`provenance/source-manifest.json`](../provenance/source-manifest.json).

| Path in archive | SHA-256 |
|---|---|
| `benchmarks/duration-calibration-20261015/run_calibration.py` | `71c9879b8e5fb25158a498d64bbd2395cc62d06baae10b55c9a2a2f893bd5702` |
| `benchmarks/minimal-generalization-20261015/analyze_study.py` | `32dc80517320c2d6fe620b1c37211b55715afed93baa3c901e67506cd941e908` |
| `benchmarks/minimal-generalization-20261015/protocol.json` | `5f97e38b0ac7ab5d3e004c85ae1b24029d133fa64a22e3a08c4e7880b2d0ff68` |
| `benchmarks/minimal-generalization-20261015/run_study.py` | `c15591c23e594309784adca6587806aeb51191efb302d7288269cacf330c44b4` |
| `docs/MINIMAL_GENERALIZATION.md` | `c3d0b7548c03ca7945ba0344f453b50a7379c8c67ce40b70937e67167ca34b3f` |
| `pyproject.toml` | `6c915693aa48f32d7492c7603d84d8a46f1dc2a8d8e097a7042331bbe94f5218` |
| `src/recurrent_cnn/__init__.py` | `4b9608fa2fbec6405399173629d04e3157b3cb706be4ce423a80e2d31503adb7` |
| `src/recurrent_cnn/__main__.py` | `985e653dcb45b624289ae78f3235686e618caa610371d3765ab89b13efa0e225` |
| `src/recurrent_cnn/analysis.py` | `07fa2763f0b62584dae3649411214f442e39affdf93a1cb3c1289eff62c3aa15` |
| `src/recurrent_cnn/backend.py` | `a8c581d2cff092114f02172995f9193b89b653efcdd0f0b7d13e135865a015a6` |
| `src/recurrent_cnn/cli.py` | `5fea043414029971c5f0789c8ac7245879701ef957e71fbf34be014f32fa5f90` |
| `src/recurrent_cnn/config.py` | `9db693b35d146643ea1381c28d037e3034a67f5d098c22cb38bb85d3ff09d123` |
| `src/recurrent_cnn/costs.py` | `463c7506bb4d35f2f4e32098c9e00d3c69bf1be608a9f5b7ac2295e106310ce4` |
| `src/recurrent_cnn/data.py` | `7cf4a43db7d367bf41da6e4a63976818c022d7eae0efee796ac0bf8f097888d5` |
| `src/recurrent_cnn/engine.py` | `ee7d436524a633b05b24504117c97f78d81bd6adbc7888402a7738798bb8d010` |
| `src/recurrent_cnn/models.py` | `0b60302bc04f885000fbecd4313907dc7903295fcdf5fa62d57185bc2f7e8dd5` |
| `src/recurrent_cnn/profiling.py` | `4240a9d40a1ad188155360747d25c145b680278bd09fba7b09afd61b0ecee594` |
| `src/recurrent_cnn/protocol.py` | `4cb28400c7050d420b5639532f8ee0e7021996b2072ca267e7777ebd12962880` |

Original scientific identifiers (package name `recurrent_cnn`, command `recurrent-cnn`, file and directory names, and the experiment identifier) are kept verbatim in the source and evidence.

## Differences between the archive and the working tree

The working tree is the public presentation of the archive. Packaging and documentation changed; scientific evidence did not.

| File | Relation to the archive |
|---|---|
| `docs/MINIMAL_GENERALIZATION.md` | Replaced by a new English protocol note. The original-language note remains in the archive, hash `c3d0b7548c03ca7945ba0344f453b50a7379c8c67ce40b70937e67167ca34b3f`. |
| `pyproject.toml` | Project name and description updated for the public repository. Current SHA-256 `e37fee9660241cbddb202bc2e5ebe49d4834e088678a41c7dcb92af4c0d66f7f`; the archived original is `6c915693aa48f32d7492c7603d84d8a46f1dc2a8d8e097a7042331bbe94f5218`. |
| `benchmarks/minimal-generalization-20261015/run_study.py` | The working tree holds the operator that completed the study (`975bed54c7f23312a157e1e009a955a7cca988be98422d649a683629f8e19d1a`); the archive holds the initial operator (`c15591c23e594309784adca6587806aeb51191efb302d7288269cacf330c44b4`). See the resumption note below. |
| All other 15 archived files | Byte-identical to the archive. |

New presentation files (`README.md`, `assets/`, `docs/`, `scripts/`) are separately committed and do not alter the benchmark evidence. The packaging name and the presentation are separate from the frozen evidence.

## Resumption of one run

Run `C-s53` was interrupted and resumed from its saved update 7,999 (next update 8,000, horizon 10,000). [`resumption.json`](../benchmarks/minimal-generalization-20261015/resumption.json) records the original and current operator hashes above, the partial artifact hashes and the log offset; [`freeze.json`](../benchmarks/minimal-generalization-20261015/freeze.json) records both operator hashes alongside the protocol, preregistration and resumption hashes. The resumption record names only this run.

## Evidence files

| File | SHA-256 |
|---|---|
| `benchmarks/minimal-generalization-20261015/protocol.json` | `5f97e38b0ac7ab5d3e004c85ae1b24029d133fa64a22e3a08c4e7880b2d0ff68` |
| `benchmarks/minimal-generalization-20261015/preregistration.json` | `cdaea8ae86a41783fedad58709486877f8cb9e76f0e782bba2a749ddf6ada7ab` |
| `benchmarks/minimal-generalization-20261015/resumption.json` | `a066f342200db8be9953391518e21ddea9392f896c47b89472eba04d4a93ea87` |
| `benchmarks/minimal-generalization-20261015/freeze.json` | `64da1d811fb71a1913042d50efc62fdda49d240b70e2f54c375ebbe328810bd0` |
| `benchmarks/minimal-generalization-20261015/validation.json` | `5d5325b3893e84c8cabc5e1c8da387c91e86e46ea5588f8bbdaa363ead2e46d7` |
| `benchmarks/minimal-generalization-20261015/test_metrics.json` | `16c6f5dc258d76c2a9d15eb5ac09a62c87e2288cab0736d63f801b2d85b13e76` |

The manifest also lists the SHA-256 of every exported file from the original, including per-run metadata and metric logs.

## Completed experiment and historical plans

The completed experiment is the exact-15-run minimal generalization study (B8, C8, E8; five paired seeds) described in [`MINIMAL_GENERALIZATION.md`](MINIMAL_GENERALIZATION.md). The source tree also carries tooling and presets written for broader model and depth matrices. Those plans were not executed for this study, and the repository includes no results from them.

## Intentional exclusions

- Private Git history, working notes, automation configuration and operational run narratives, including the original root report. [`RESULTS.md`](RESULTS.md) is the concise results authority for this export.
- Datasets (CIFAR-100 is not redistributed) and model checkpoints. Checkpoint SHA-256 hashes remain in the evidence files.

## Verifying the archive

```bash
shasum -a 256 provenance/frozen-source.zip

python - <<'PY'
import hashlib, json, zipfile

manifest = json.load(open("provenance/source-manifest.json"))["original_source_sha256"]
archive = zipfile.ZipFile("provenance/frozen-source.zip")
for name, expected in sorted(manifest.items()):
    status = "ok" if hashlib.sha256(archive.read(name)).hexdigest() == expected else "MISMATCH"
    print(status, name)
PY
```

## Authorship

This repository asserts no author, affiliation, publication or DOI. To cite it, use the repository URL and the exact commit hash, as described in the [README](../README.md#citation).
