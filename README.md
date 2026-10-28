# FCR-AD

**Forecasting Context Reuse for Anomaly Detection**

Core implementation of *FCR-AD: Semi-Supervised Anomaly Detection for Asynchronous UAV Telemetry via Forecasting Context Reuse*.

FCR-AD learns observation forecasts from healthy UAV telemetry. The same historical channel contexts support query-time prediction and fault classification. After forecasting is frozen, GradBoost combines prediction residuals with a pooled historical representation.

```text
native asynchronous history
    → time-aware patch aggregation
    → causal temporal convolution
    → Patch Pooling → channel contexts → query-time forecasts
                          ↓                     ↓ real evidence
                    Context Summary         residuals
                          └──────────┬──────────┘
                                  GradBoost
                                     ↓
                         calibrated anomaly score / decision
```

## Install and check

Python 3.10 or newer is required. The reference environment uses Python 3.10.20, NumPy 1.26.4, pandas 2.3.3, PyTorch 2.6.0 and scikit-learn 1.7.2.

```bash
git clone https://github.com/ZCchou/FCR-AD.git
cd FCR-AD
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python -m fcr_ad smoke
```

The smoke command runs one gradient update, feature extraction, classifier fitting, calibration, and evaluation on synthetic inputs. Install the appropriate PyTorch build for your CPU or CUDA platform.

## ALFA data

Obtain the processed topic CSVs from the [ALFA dataset](https://theairlab.org/alfa-dataset/). Each flight directory contains timestamped navigation topics, for example `*-mavros-nav_info-airspeed.csv`, and fault-status topics where applicable.

```bash
python -m fcr_ad prepare --source-root /path/to/ALFA/processed --output data/alfa
python -m fcr_ad check-data --data-root data/alfa
```

Preparation outer-joins the six navigation topics at their original nanosecond timestamps, converts times to seconds relative to the recorded flight origin, and preserves empty observations. It writes the 17-channel CSV representation, including yaw; the model loader selects the retained 15 channels. No interpolation or observation filling is applied. The first 20 seconds are excluded when windows are loaded, without resetting the flight time reference.

The published metadata fixes 25 training, eight validation and 13 test files. Event timing is read from that metadata. The prepare command checks fault onset against the source status messages and verifies the exported feature hashes. It writes feature CSVs and event metadata; separate label CSVs are not needed by this loader. Original telemetry and trained checkpoints are not included in this repository.

An existing processed data root can also be used directly:

```text
alfa/
  metadata/file_manifest.csv
  train/csv/<flight>.csv
  val/csv/<flight>.csv
  test/csv/<flight>.csv
```

Feature CSVs contain `t` and the channel columns in `metadata/feature_mapping.json`. Missing channel observations remain empty. The loader verifies file assignment, content hashes, complete observation capacity, and the expected window counts:

| Role | Windows | Anomalous windows |
| --- | ---: | ---: |
| Healthy training | 1475 | 0 |
| Healthy validation | 386 | 0 |
| Classifier training | 1928 | 447 |
| Classifier validation | 489 | 101 |
| Test | 1123 | 227 |

## Train and evaluate

```bash
python -m fcr_ad run --data-root data/alfa --output runs/fcr_ad --seeds 42 43 44 --device cuda
```

Use `--device cpu` on a CPU machine. A single seed is the default. Training and evaluation can also be separated:

```bash
python -m fcr_ad train --data-root data/alfa --output runs/fcr_ad --seeds 42
python -m fcr_ad evaluate --data-root data/alfa --output runs/fcr_ad --seeds 42
```

`configs/alfa_author_fixed.json` records the released protocol: 5-second history, 1-second evidence interval, 1-second stride, four patches, healthy-window AdamW training, and a 290-feature classifier. Healthy validation MSE selects the backbone. Labeled validation windows select a threshold that maximizes Raw F1 with normal-window FPR at most 5%; ties prefer lower FPR, then a higher threshold. All decisions use `score > threshold`.

Training resumes from the most recent validation checkpoint when the run identity is unchanged. A frozen run records the backbone, classifier, normalization, protocol, calibration and source hashes. Evaluation verifies these artifacts before loading test windows.

Each seed directory contains the forecasting checkpoint, classifier, training history, calibration and `frozen_before_test.json`. Test outputs are in `evaluation/metrics.json` and `evaluation/predictions.csv`. Raw scores support sequential window decisions. PA F1 uses annotated event boundaries and is an offline evaluation measure.

## Core API

```python
from fcr_ad.model import make_backbone
from fcr_ad.features import extract_features
from fcr_ad.head import Head
from fcr_ad.evaluation import choose_threshold

model = make_backbone()
# The five inputs are padded [batch, channel, observation] tensors.
prediction, channel_context = model.forward_with_context(
    hist_times, hist_values, hist_mask, query_times, query_mask
)
features = extract_features(model, window_dataset)
# features: 34 residual statistics + 256 pooled historical features.
head = Head().fit(training_features, training_rows, seed=42)
calibration = choose_threshold(validation_labels, head.predict(validation_features))
```

## Repository contents

- `fcr_ad/`: model, native data preparation/windowing, feature extraction, GradBoost, training, calibration and evaluation.
- `configs/`: fixed main-method protocol.
- `splits/` and `metadata/`: flight membership, topic mapping and source feature checksums.
- `tests/`: model/data contracts, threshold and event handling, serialization, and training/freeze checks.
- `docs/implementation.md`: tensor interfaces, feature order and checkpoint compatibility.


