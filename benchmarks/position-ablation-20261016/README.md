# Controlled position-of-untying follow-up

This separate exploratory experiment asks whether untied **boundary** operators help more than the same number of untied **interior** operators. Both maps use three independent branches, identical 1/6/1 use counts, 59,844 parameters and 37,973,120 counted forward MACs. All branches start as copies of the same native C-initialized branch, giving both arms the same initial function.

- **END:** `[A,S,S,S,S,S,S,Z]`.
- **INNER:** `[S,A,S,S,S,S,Z,S]`.

The original B8/C8/H8 results were already known when this question was specified. This is not independent confirmation. New END is **not** the historical H8 initialization. The primary comparison uses five paired seeds and the existing validation split; that split is also used for checkpoint selection, so results are conditional exploratory estimates.

The frozen recipe, endpoints, diagnostics and preservation boundaries are specified in [`protocol.json`](protocol.json). There is no official-test evaluation stage, historical control retraining or outcome-dependent tuning.

## Completed validation-only result

The five paired seeds (11, 23, 37, 53, 71) completed ten runs of 10,000 updates each (100,000 final logical study updates). END selected-checkpoint validation top-1 was **21.672 ± 0.5168%** (mean ± sample SD); INNER was **21.096 ± 0.6520%**. The paired END−INNER contrast was **+0.576 pp** (nominal, unadjusted two-sided Student-t 95% CI **[+0.2194, +0.9326]**, df=4; 5/5 seeds positive).

This is conditional exploratory evidence favoring boundary placement under this recipe—not independent confirmation, universal evidence, or official-test evidence. The selected validation split (seed 1729) also selected checkpoints; paired seeds, not repeated images, are the inference units. END did not win across metrics: mean selected cross-entropy was 3.547269 vs 3.527459 (END−INNER +0.0198106; nominal CI [−0.004803, +0.044424], crossing zero), and mean 15-bin ECE was 13.287063% vs 11.767394% (+1.519669 pp; lower is better). The larger END train/validation gap is descriptive, not proof of overfitting.

The generated [`analysis/MECHANISM_ABLATION.md`](analysis/MECHANISM_ABLATION.md) is the authoritative detailed report; [`analysis/summary.json`](analysis/summary.json), [`analysis/contrasts.csv`](analysis/contrasts.csv), and [`analysis/table.csv`](analysis/table.csv) are the machine-readable evidence. The README tables/report text are display summaries. Provenance and continuation details are in [`provenance/README.md`](provenance/README.md). Historical B/C/H results and freezes are unchanged; new END uses cloned C-initialized branches and is not historical H8.

### Offline analysis regeneration

From the repository root, choose a new output path that does not already exist:

```bash
python benchmarks/position-ablation-20261016/analyze_position.py \
  --validation benchmarks/position-ablation-20261016/validation.json \
  --runs benchmarks/position-ablation-20261016/runs \
  --control-validation benchmarks/minimal-generalization-20261015/validation.json \
  --diagnostics benchmarks/position-ablation-20261016/diagnostics/diagnostics.json \
  --output NEW_ANALYSIS_DIR
```

This deterministic scalar analysis reads committed JSON and compressed scalar-log evidence. Reproducing CUDA gradient diagnostics is separate and requires the bound original checkpoints and data; model weights and data are not distributed.
