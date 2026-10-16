| Model | Test top-1 at K8, mean (95% CI) | Per-seed test top-1 at K8 | Mean cross-entropy (native units) | Parameters | MACs / inference |
|:--|--:|:--|--:|--:|--:|
| Untied baseline (B) | 18.39% (17.58–19.20) | 19.22, 17.86, 18.26, 18.89, 17.71 | 3.631 | 152,644 | 37,973,120 |
| Tied recurrent model (C) | 20.46% (19.69–21.23) | 19.59, 20.86, 20.79, 20.03, 21.03 | 3.546 | 22,724 | 37,973,120 |
| Recurrent + recall, variable depth, progressive loss (E) | 15.67% (14.39–16.95) | 17.50, 15.44, 15.11, 15.23, 15.09 | 3.691 | 24,772 | 42,167,424 |
| Endpoint-independent, shared-center hybrid (H8, exploratory follow-up) | 21.28% (20.62–21.95) | 20.95, 21.01, 21.84, 21.88, 20.74 | 3.562 | 59,844 | 37,973,120 |

Paired C8−B8 test difference: **2.07 percentage points** (95% paired Student-t CI 0.50–3.64).

Per-seed values are ordered by seeds 11, 23, 37, 53, 71. Parameter counts are unique parameters; B and C have equal reported MACs per inference. Fewer parameters do not imply lower compute. MAC estimates omit normalization, activations, residual operations, pooling, and memory traffic.

H8 exploratory follow-up: H8−C8 was +0.82 pp (nominal 95% CI −0.27 to 1.91; unresolved) and H8−B8 was +2.90 pp (2.04 to 3.76). H8 has 2.63× C8's parameters, confounding attribution. H8 was evaluated after the original official-test results were known, is outside the original protocol/hierarchy, is not independent confirmation, and leaves the primary C8−B8 result unchanged.
