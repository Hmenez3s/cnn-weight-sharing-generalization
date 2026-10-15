# Safe reproduction entrypoints

These entrypoints separate a disposable synthetic software check from the guarded native study. They do not alter the research implementation, acquire data, access the network, or start scientific training.

## Synthetic CPU smoke

From the project root, run:

```sh
python scripts/smoke.py
```

This creates a unique temporary root and private cache directories, calls the existing module CLI with synthetic preparation (`--per-class 3`, paired seed 0), then runs the existing smoke recipe for exactly two CPU updates. It reports `scientific: false`; temporary artifacts are removed when the process exits. No GPU or CIFAR data is used.

## Prepare a separate native-study root

Provide a **local, read-only CIFAR-100 training-data location** (the directory containing `train` and `meta`, or its parent containing `cifar-100-python`) and a new output path whose final directory does not exist:

```sh
python scripts/prepare_replication.py \
  --root /absolute/path/to/new-replication-root \
  --training-root /absolute/path/to/local/cifar-100
```

Preparation creates the destination exclusively, snapshots the source set listed by `provenance/source-manifest.json`, records new SHA-256 values for the actual snapshot separately from historical `original_source_sha256` values, copies the protocol, and verifies the expected official checksums for the local `train` and `meta` files and their copies. Only those two files are copied. The input is never modified; the official `test` file is never read or copied. A missing source file (including the published protocol document required by the source manifest) is an error. Never reuse an existing root.

The command prints shell setup and exact native-operator train/evaluate invocations. Use the generated root and retain every private environment path under `$ROOT/private`. The native operator still requires the specifically confirmed **RTX 5090**, 12 CPU threads, and a 50% per-process VRAM cap; this is not a claim of general GPU compatibility. Do not run these operator phases as part of a smoke test.

The printed test-data instruction is manual and gated: do not copy or open official test data until all 15 training runs are complete and their checkpoints/artifacts have been independently verified frozen, with evaluation authorized. The preparation tool neither performs nor authorizes that later action. Full replication status remains **NOT RUN**. Independent reanalysis requires only the published JSON results and does not require this root or local CIFAR data.
