# Controlled position-of-untying follow-up

This separate exploratory experiment asks whether untied **boundary** operators help more than the same number of untied **interior** operators. Both maps use three independent branches, identical 1/6/1 use counts, 59,844 parameters and 37,973,120 counted forward MACs. All branches start as copies of the same native C-initialized branch, giving both arms the same initial function.

- **END:** `[A,S,S,S,S,S,S,Z]`.
- **INNER:** `[S,A,S,S,S,S,Z,S]`.

The original B8/C8/H8 results were already known when this question was specified. This is not independent confirmation. New END is **not** the historical H8 initialization. The primary comparison uses five paired seeds and the existing validation split; that split is also used for checkpoint selection, so results are conditional exploratory estimates.

The exact recipe, endpoints, diagnostics, limitations and preservation boundaries are specified in [`protocol.json`](protocol.json). There is no official-test evaluation stage, no historical control retraining and no outcome-dependent tuning. Ten runs of 10,000 updates are authorized. Historical studies remain unchanged. Results and execution evidence will be added only after the pre-specified run completes.
