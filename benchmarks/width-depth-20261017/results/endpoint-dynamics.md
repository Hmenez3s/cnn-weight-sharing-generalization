# Endpoint dynamics: exploratory posthoc diagnostics

Official-test access: false. Paired Student-t 95% intervals use df=4 and are nominal.
First crossings are noisy sampled validation observations, not wall-clock or causal stopping-time evidence.

## Selected and final metrics

Selected CE uses the archived analysis checkpoint value; matched-update and final CE use raw training logs.
Maximum observed absolute archive-versus-training-log selected-CE difference: C8=5.49316405696e-08; H8=9.76562501975e-08; C7B1=0; C4B4=0; B4C4=0

| Model | Selected update | Selected train | Selected val | Selected gap | Selected CE (archive) | Final train | Final val | Final gap | Final CE (log) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C8 | 9080.000 ± 1073.313 | 38.218 ± 1.383 | 21.252 ± 0.406 | 16.966 ± 1.598 | 3.529 ± 0.050 | 38.782 ± 0.982 | 21.172 ± 0.407 | 17.610 ± 1.254 | 3.534 ± 0.044 |
| H8 | 8440.000 ± 804.984 | 41.391 ± 2.266 | 21.748 ± 1.254 | 19.643 ± 2.143 | 3.536 ± 0.080 | 42.884 ± 1.008 | 21.528 ± 1.290 | 21.356 ± 0.684 | 3.572 ± 0.069 |
| C7B1 | 9200.000 ± 989.949 | 37.733 ± 0.802 | 21.360 ± 0.460 | 16.373 ± 0.800 | 3.455 ± 0.023 | 38.160 ± 0.293 | 21.196 ± 0.569 | 16.964 ± 0.615 | 3.460 ± 0.027 |
| C4B4 | 7840.000 ± 1013.903 | 32.804 ± 0.977 | 20.640 ± 0.517 | 12.164 ± 1.204 | 3.409 ± 0.032 | 34.080 ± 0.912 | 20.424 ± 0.530 | 13.656 ± 1.298 | 3.424 ± 0.030 |
| B4C4 | 7000.000 ± 1303.840 | 45.973 ± 6.936 | 20.888 ± 0.287 | 25.085 ± 6.831 | 3.768 ± 0.220 | 56.036 ± 1.016 | 20.108 ± 0.366 | 35.928 ± 1.023 | 4.145 ± 0.071 |

## Paired contrasts (left minus right)

| Contrast | Phase | Metric | Mean difference | 95% CI | + / - / ties |
|---|---|---|---:|---:|---:|
| H8-C7B1 | selected | selected_update | -760.000 | [-2158.182, 638.182] | 1/4/0 |
| H8-C7B1 | selected | clean_train_accuracy_percent | 3.658 | [0.776, 6.540] | 5/0/0 |
| H8-C7B1 | selected | validation_accuracy_percent | 0.388 | [-1.148, 1.924] | 3/1/1 |
| H8-C7B1 | selected | gap_pp | 3.270 | [0.778, 5.761] | 5/0/0 |
| H8-C7B1 | selected | validation_loss | 0.080 | [0.000, 0.160] | 5/0/0 |
| H8-C7B1 | final | clean_train_accuracy_percent | 4.724 | [3.649, 5.800] | 5/0/0 |
| H8-C7B1 | final | validation_accuracy_percent | 0.332 | [-1.230, 1.894] | 2/3/0 |
| H8-C7B1 | final | gap_pp | 4.392 | [3.809, 4.975] | 5/0/0 |
| H8-C7B1 | final | validation_loss | 0.112 | [0.035, 0.189] | 5/0/0 |
| B4C4-C4B4 | selected | selected_update | -840.000 | [-1808.179, 128.179] | 1/4/0 |
| B4C4-C4B4 | selected | clean_train_accuracy_percent | 13.169 | [4.698, 21.640] | 5/0/0 |
| B4C4-C4B4 | selected | validation_accuracy_percent | 0.248 | [-0.489, 0.985] | 3/2/0 |
| B4C4-C4B4 | selected | gap_pp | 12.921 | [4.352, 21.490] | 5/0/0 |
| B4C4-C4B4 | selected | validation_loss | 0.359 | [0.080, 0.637] | 5/0/0 |
| B4C4-C4B4 | final | clean_train_accuracy_percent | 21.956 | [19.695, 24.216] | 5/0/0 |
| B4C4-C4B4 | final | validation_accuracy_percent | -0.316 | [-1.073, 0.441] | 1/4/0 |
| B4C4-C4B4 | final | gap_pp | 22.272 | [19.447, 25.096] | 5/0/0 |
| B4C4-C4B4 | final | validation_loss | 0.721 | [0.633, 0.808] | 5/0/0 |

## First validation threshold crossings

Means are conditional on seeds reaching the threshold.

| Model | Threshold (%) | Attained | Mean update among attainers |
|---|---:|---:|---:|
| C8 | 18 | 5/5 | 3560.0 |
| C8 | 19 | 5/5 | 4760.0 |
| C8 | 20 | 5/5 | 5960.0 |
| H8 | 18 | 5/5 | 3800.0 |
| H8 | 19 | 5/5 | 4520.0 |
| H8 | 20 | 4/5 | 4800.0 |
| C7B1 | 18 | 5/5 | 4000.0 |
| C7B1 | 19 | 5/5 | 4520.0 |
| C7B1 | 20 | 5/5 | 5600.0 |
| C4B4 | 18 | 5/5 | 4440.0 |
| C4B4 | 19 | 5/5 | 5320.0 |
| C4B4 | 20 | 4/5 | 6650.0 |
| B4C4 | 18 | 5/5 | 3880.0 |
| B4C4 | 19 | 5/5 | 4880.0 |
| B4C4 | 20 | 5/5 | 5720.0 |

## Matched updates

CE is from raw training logs.

| Model | Update | Train (%) | Validation (%) | Gap (pp) | Validation CE (log) |
|---|---:|---:|---:|---:|---:|
| C8 | 2000 | 18.560 | 14.504 | 4.056 | 3.6281 |
| C8 | 4000 | 26.289 | 17.784 | 8.505 | 3.5084 |
| C8 | 6000 | 33.458 | 20.148 | 13.310 | 3.4809 |
| C8 | 8000 | 37.356 | 20.640 | 16.716 | 3.5252 |
| C8 | 10000 | 38.782 | 21.172 | 17.610 | 3.5339 |
| H8 | 2000 | 17.609 | 13.708 | 3.901 | 3.6437 |
| H8 | 4000 | 27.107 | 18.524 | 8.583 | 3.4424 |
| H8 | 6000 | 34.320 | 20.216 | 14.104 | 3.4739 |
| H8 | 8000 | 40.613 | 21.164 | 19.449 | 3.5426 |
| H8 | 10000 | 42.884 | 21.528 | 21.356 | 3.5718 |
| C7B1 | 2000 | 17.969 | 14.184 | 3.785 | 3.6353 |
| C7B1 | 4000 | 26.320 | 18.108 | 8.212 | 3.4689 |
| C7B1 | 6000 | 32.938 | 20.080 | 12.858 | 3.4358 |
| C7B1 | 8000 | 37.142 | 20.884 | 16.258 | 3.4484 |
| C7B1 | 10000 | 38.160 | 21.196 | 16.964 | 3.4600 |
| C4B4 | 2000 | 15.707 | 13.044 | 2.663 | 3.6848 |
| C4B4 | 4000 | 23.271 | 16.956 | 6.315 | 3.5228 |
| C4B4 | 6000 | 29.316 | 19.280 | 10.036 | 3.4401 |
| C4B4 | 8000 | 32.707 | 20.216 | 12.491 | 3.4373 |
| C4B4 | 10000 | 34.080 | 20.424 | 13.656 | 3.4244 |
| B4C4 | 2000 | 16.756 | 13.284 | 3.472 | 3.6586 |
| B4C4 | 4000 | 28.044 | 17.848 | 10.196 | 3.5030 |
| B4C4 | 6000 | 38.662 | 19.672 | 18.990 | 3.6454 |
| B4C4 | 8000 | 50.196 | 20.036 | 30.160 | 3.9674 |
| B4C4 | 10000 | 56.036 | 20.108 | 35.928 | 4.1451 |
