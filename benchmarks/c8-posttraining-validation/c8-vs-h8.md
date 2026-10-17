# C8 vs H8 post-training comparison

Descriptive validation-only comparison; no causal or significance claims. Mean-per-seed methods compare the unweighted mean of five per-seed metric rows. The five-seed ensemble is one pooled probability predictor.

| Method | Passes/image | Accuracy C8 % | H8 % | C8−H8 pp | ECE C8 % | H8 % | C8−H8 pp | NLL C8 / H8 | Brier C8 / H8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 21.256 | 21.744 | -0.488 | 12.369 | 12.987 | -0.618 | 3.52857 / 3.53555 | 0.92354 / 0.92419 |
| temperature scaling | 1 | 21.256 | 21.744 | -0.488 | 2.806 | 3.470 | -0.664 | 3.29965 / 3.27023 | 0.90079 / 0.89952 |
| weight average | 1 | 21.200 | 21.720 | -0.520 | 12.638 | 13.452 | -0.814 | 3.52890 / 3.54631 | 0.92358 / 0.92507 |
| five seed ensemble | 5 | 28.460 | 29.320 | -0.860 | 4.791 | 5.166 | -0.376 | 2.91171 / 2.84724 | 0.84553 / 0.83887 |
| tta | 10 | 21.592 | 21.652 | -0.060 | 6.422 | 7.274 | -0.853 | 3.38268 / 3.37514 | 0.90508 / 0.90426 |
| all four combined | 50 | 29.160 | 29.880 | -0.720 | 4.081 | 4.193 | -0.112 | 2.88956 / 2.82667 | 0.84318 / 0.83448 |

C8 has 22,724 parameters and H8 has 59,844; both have 37,973,120 MACs per view. Five-seed ensembles store/use five models; costs are arithmetic counts, not measured speed. This report makes no causal or statistical-significance claim.
