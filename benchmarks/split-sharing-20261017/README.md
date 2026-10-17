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
