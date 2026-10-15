# Distribution and third-party notices

The original software and new presentation in this repository are distributed under [Apache-2.0](LICENSE). External dependencies retain their own licenses; no dependency source trees or binary wheels are vendored here.

## Direct dependencies

The license texts below were inspected from their official repositories. Exact source URLs and fetched-text SHA-256 digests are recorded in [the license evidence](provenance/dependency-licenses.json). This is a bounded dependency review, not a legal opinion or an exhaustive audit of every optional/transitive binary component.

| Dependency | License | Source |
|---|---|---|
| PyTorch | BSD-3-Clause | [Official license](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/LICENSE) |
| TorchVision | BSD-3-Clause | [Official license](https://raw.githubusercontent.com/pytorch/vision/v0.24.1/LICENSE) |
| NumPy | BSD-3-Clause | [Official license](https://raw.githubusercontent.com/numpy/numpy/main/LICENSE.txt) |
| SciPy | BSD-3-Clause | [Official license](https://raw.githubusercontent.com/scipy/scipy/main/LICENSE.txt) |
| Matplotlib | PSF-based Matplotlib license | [Official license](https://raw.githubusercontent.com/matplotlib/matplotlib/main/LICENSE/LICENSE) |
| PyYAML | MIT | [Official license](https://raw.githubusercontent.com/yaml/pyyaml/main/LICENSE) |
| pytest | MIT | [Official license](https://raw.githubusercontent.com/pytest-dev/pytest/main/LICENSE) |

## Dataset and checkpoints

CIFAR-100 images and the source dataset archive are **not redistributed**. Obtain data separately from the [dataset's official page](https://www.cs.toronto.edu/~kriz/cifar.html), review the applicable rights and restrictions, and cite Alex Krizhevsky's 2009 technical report, *Learning Multiple Layers of Features from Tiny Images*. The project license does not grant rights to the dataset. Published measurements, split indices, hashes and training logs do not include image pixels.

No trained checkpoint is included in this repository or promised as a public download. Original checkpoints are retained privately for recovery; redistribution rights and storage would need to be resolved before a separate checkpoint release. The small ZIP under `provenance/` contains source text, not model weights or data.

## Attribution and scope

The implementation uses dependency APIs and credits related scientific work in the README. It does not claim a new invention, a state-of-the-art result, or a published paper. No author, affiliation or DOI is asserted. Repository citation guidance uses the repository URL and a verified release or commit; a formal authorship record has not been established.

Private operational records are excluded from the public export. Original scientific source and measurements are retained verbatim where archived; presentation and packaging changes are separate from those frozen artifacts.
