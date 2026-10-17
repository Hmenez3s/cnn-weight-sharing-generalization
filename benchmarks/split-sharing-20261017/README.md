# Split-sharing study (validation-only)

Exploratory follow-up locating sharing in an eight-update trunk. C4B4 uses
`(0,0,0,0,1,2,3,4)`; B4C4 uses `(0,1,2,3,4,4,4,4)`. Both retain five
independently initialized branches, 96,964 parameters, and 37,973,120 MACs/image.
Initialization constructs native B8 and retains its first five branches; the
family-independent derived init seed therefore also matches paired C8 stems,
heads, and common branch 0.

The copied B8/C8/H8 recipe uses 10% of training data, paired seeds 11/23/37/53/71,
10,000 updates, batch 128, SGD lr .03/momentum .9/weight decay .0005, cosine
10,000, validation every 200 updates, and `amended_v2` selection. Historical
training identities are checked against each `H-s<seed>/run.json` before training.
The official test partition is never opened. `--probe` uses 20 updates and marks
the cohort non-scientific; synthetic runs are also non-scientific. Set
`--workers N` for concurrent isolated child processes; `workers × memory-fraction`
must not exceed .9. Controls are reused from committed B8/C8 validation tables
and H8 validation tables/logs. This is validation-only and exploratory; no
official-test inference or confirmatory claim is part of this study.

Example: `python benchmarks/split-sharing-20261017/run_split.py --data-root DATA --output OUT --workers 2`

Analyze a complete five-seed cohort with:
`python benchmarks/split-sharing-20261017/analyze_split.py --cohort OUT/cohort.json --output ANALYSIS`

## Results

The five-seed cohort ran on one RTX 5090 with five concurrent worker processes
(two threads each): 10 runs in 1,228 s. A matched 400-update probe measured 2.2× the sequential throughput; against the original ≈ 375 s per sequential run the full cohort was ≈ 3× faster. The
committed B8/C8/H8 control means reproduce the published values. Repeated
sequential probes show last-digit floating-point differences of the same size
as parallel versus sequential runs, so concurrency does not change results
beyond normal GPU nondeterminism.

| Model | Validation accuracy | Validation CE | Train, selected | Train, end | Gap, end |
|---|---:|---:|---:|---:|---:|
| B8 | 18.80 ± 0.64 | 3.59 | 35.5 | 41.4 | 23.0 |
| C8 | 21.25 ± 0.41 | 3.53 | 38.2 | 38.8 | 17.6 |
| H8 | 21.75 ± 1.25 | 3.54 | 41.4 | 42.9 | 21.4 |
| C4B4 | 20.64 ± 0.52 | 3.41 | 32.8 | 34.1 | 13.7 |
| B4C4 | 20.89 ± 0.29 | 3.77 | 46.0 | 56.0 | 35.9 |

Per-seed values and paired contrasts: [`results/analysis.md`](results/analysis.md)
([JSON](results/analysis.json)). Cohort manifest: [`results/cohort.json`](results/cohort.json);
per-run `run.json` and compressed training logs in [`results/runs/`](results/runs/).
Checkpoints are not distributed. Figure: `scripts/make_split_sharing_figure.py`.

## Optional ordered arms

`--arms C7B1 C4W45` selects any unique ordered subset of `C4B4 B4C4 C7B1 C4W45`; omitting it preserves the historical two-arm cohort. Runs are scheduled seed-major, with arms in requested order. C7B1 uses map `(0,0,0,0,0,0,0,1)` and two branches retained from native B8 initialization: source branches `[1,2]`, matching H8's shared S and final Z branches while replacing only H8's first A update. It has 41,284 parameters and 37,973,120 MAC/image. C4W45 is native family C, depth 4, width 45, with one shared block and 42,445 parameters; its 37,640,340 MAC/image is approximately (not exactly) matched to the C8 forward-MAC target (−0.8764%). Native GroupNorm uses one group at width 45 versus eight at width 32, so depth, width, and normalization grouping jointly vary. This is exploratory validation-only work; paired nominal intervals span multiple comparisons, and historical B8/C8/H8 controls are reused rather than retrained.
