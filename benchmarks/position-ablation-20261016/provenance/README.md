# Position-ablation provenance and continuation

## Bound study and prospective record

This documentation describes provenance for the validation-only END/INNER study; it is not a second results report. The detailed generated analysis is [`../analysis/MECHANISM_ABLATION.md`](../analysis/MECHANISM_ABLATION.md), with authoritative machine-readable outputs in [`../analysis/summary.json`](../analysis/summary.json) and the related CSVs.

The original prospective preregistration is [`prospective-preregistration.json`](prospective-preregistration.json), SHA-256 `8b5fad1827fc841677b4d84795ccd3ace62b5382f25cc203fbdacb75d840c15c`. The root [`../preregistration.json`](../preregistration.json) is regenerated administrative identity, not a new prospective preregistration. The continuation receipt records both identities and hashes, confirms the prior bound inputs and source hashes were unchanged, and records no outcome-dependent decisions. See [`continuation_receipt.json`](continuation_receipt.json) and [`continuation-input.json`](continuation-input.json) for exact bound hashes and artifact lists.

The frozen scientific source head is `320ae7dbfd512db4648d6adcd76ea9bb1e5312fa`; continuation adapter head `fc028965bb7a67545589ae2b64aa1d8073b5d00f`; diagnostics source `3a5daa599b78ccad90f63a3296cc95945b4ee1d7`; analyzer `8bdad32a52e10db2b9e516ff5fbba40b2704b8df`. The frozen scientific source, recipe, protocol and tolerances were unchanged.

## Continuation accounting

Five paired seeds (11, 23, 37, 53, 71), ten runs of 10,000 updates, produced 100,000 final logical study updates. Two complete prior runs (END-s11 and INNER-s11) were reused. END-s23 resumed from checkpoint update 3,202; seven runs were new. The continuation invocation performed 76,798 updates. One previously committed but uncheckpointed update was replayed. The receipt claims **at least 100,001 optimizer updates across attempts**, not an exact physical total; the amount of prior uncheckpointed physical work is unknown. No exact total physical-update count is claimed.

## Runtime and validation evidence

The CUDA diagnostic producer completed all five seeds with reported guard counters at zero; all 20 selected checkpoints (ten new arms plus ten frozen controls) met the exact top-1, cross-entropy/ECE absolute `2e-5`, and identical 15-bin-count checks. An earlier CPU replay failed the same cross-entropy gate and produced no accepted result; tolerances were not relaxed. Training completion is recorded separately in [`train_receipt.json`](../train_receipt.json); frozen run/checkpoint identities are in [`freeze.json`](../freeze.json). The accepted diagnostic aggregate, guard counters and checkpoint bindings are in [`diagnostics/diagnostics.json`](../diagnostics/diagnostics.json).

Recorded runtime was RTX 5090, PyTorch 2.9.1+cu128, 12 CPU threads and CUDA memory fraction 0.5. The last paired-seed completion event occurred 7.501943 seconds after diagnostic start; final hash checks and output I/O followed, so this is not end-to-end wall time. Scalar export took 0.607896 seconds with 12 compression workers; deterministic analysis took 1.966 seconds in one process.

## Interpretation and artifact boundary

There was no official-test access and no historical control retraining. END/INNER use cloned C-initialized branches; this is distinct from historical H8 initialization. The selected validation split (seed 1729) was used for selection and evaluation, so estimates are conditional and seeds are the statistical units. Gradient probes show local shared-six coherence associations only: they do not identify a causal mechanism, establish gradient conflict as a cause, or represent actual SGD/momentum/weight-decay updates.

Coordinator hash checks found all 58 protected original/reference files unchanged; the interrupted native root was preserved. These checks and reported guard counters are not an OS-level sandbox or a complete-host audit. Native JSON may use CRLF: the repository's `.gitattributes` byte-preservation policy covers nested position-study JSON, and hash-bound evidence must not be newline-normalized. No images, checkpoints, or private paths are included in the public export. Checkpoint hashes are recorded, but weights and data are not distributed. Scalar analysis can be regenerated offline from committed evidence; rerunning CUDA diagnostics requires the bound original checkpoints and data. The exact offline analysis command is in the [study README](../README.md#offline-analysis-regeneration).
