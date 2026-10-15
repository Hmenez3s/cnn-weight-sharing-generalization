| Model | Official-test top-1 mean (nominal 95% CI) | Per-seed official-test accuracy | Mean cross-entropy | Parameters | MACs/example |
|:--|--:|:--|--:|--:|--:|
| Fully independent blocks (B8) | 18.39% (17.58–19.20) | 19.22, 17.86, 18.26, 18.89, 17.71 | 3.631 | 152,644 | 37,973,120 |
| Fully shared block (C8) | 20.46% (19.69–21.23) | 19.59, 20.86, 20.79, 20.03, 21.03 | 3.546 | 22,724 | 37,973,120 |
| Endpoint-independent, central-shared blocks (H8) | 21.28% (20.62–21.95) | 20.95, 21.01, 21.84, 21.88, 20.74 | 3.562 | 59,844 | 37,973,120 |

Values are derived from validated input rows; seeds are 11, 23, 37, 53, 71. Intervals are descriptive nominal intervals across five seed-level outcomes.
