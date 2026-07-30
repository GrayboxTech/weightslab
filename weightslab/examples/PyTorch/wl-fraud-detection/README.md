# WeightsLab — Real Bank Fraud Detection (tabular, PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/wl-fraud-detection.ipynb)

A fully-runnable **tabular binary-classification** example trained on the real
[Kaggle "Credit Card Fraud Detection" dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(ULB): 284,807 European card transactions, 492 fraudulent (~0.173%
prevalence). An MLP learns to flag fraud, streaming per-sample loss /
prediction / accuracy to the WeightsLab UI. Being tabular, it's the natural
companion to the **List Exploration (tabular) view** — sort by `loss` or
`prediction` to triage the transactions the model finds hardest.

## Getting the data

There's no anonymous mirror of this dataset — pick one:

**Option A — `kagglehub` (recommended, one-time setup):**

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.login()"   # or place ~/.kaggle/kaggle.json
```

`main.py` then downloads and caches `creditcard.csv` automatically on first
run (see `utils/kaggle_data.py`).

**Option B — manual download:**

Download `creditcard.csv` from the
[Kaggle page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and
either drop it at `wl-fraud-detection/.cache/creditcard.csv`, or point
`dataset.csv_path` in `config.yaml` (or env `WL_FRAUD_CSV_PATH`) directly at
the file.

## Quick start

```bash
cd weightslab/examples/PyTorch/wl-fraud-detection
python main.py
```

Then open the UI (e.g. `http://localhost:5173`), switch the Data Exploration
board to **List** view, press play, and sort columns to explore.

## The data

Each row is one transaction: `Time` (seconds since the first transaction in
the two-day window), 28 PCA-anonymized components `V1`..`V28` (the original
authors ran PCA for confidentiality — there's no way to recover what they
represent), `Amount`, and the label `Class` (1 = fraud). `utils/data.py`
standardizes all 30 features using **training-split statistics only** (no
leakage from the test split) before feeding them to the model; raw values are
kept separately as sortable UI columns.

### Two real-world problems a synthetic dataset would hide

1. **The CSV is genuinely large (~144MB, 284,807 rows).** Re-parsing it with
   pandas on every run is wasteful. `utils/data.py` parses once (with
   `float32` dtypes — half the memory of the default `float64`) and caches the
   result as a compressed `.npz` keyed by the source file's size + mtime,
   under `dataset.cache_dir` (default: `.cache/`, already covered by the
   repo's `.gitignore`). Every subsequent run skips CSV parsing entirely.

2. **The imbalance (~1:578) is too extreme to train on directly.** A random
   batch of a few hundred rows usually contains *zero* fraud examples, so
   gradients rarely see the minority class at all. `utils/data.py` addresses
   this in the **training split only** (the test split is always left at the
   natural prevalence, so evaluation reflects reality):
   - `dataset.oversample_fraud_ratio` duplicates fraud rows (with small
     Gaussian jitter, so duplicates aren't bit-identical and the model can't
     just memorize them) up to a target ratio (default `0.1` = 10%).
   - Class weights for the loss are then **computed from the actual training
     labels** (post-oversampling) via `compute_class_weights`, not
     hardcoded — the effective ratio moves whenever you change
     `oversample_fraud_ratio` or `max_train_samples`, and a hardcoded weight
     would silently go stale. `class_weight_cap` (default `20.0`) bounds it,
     since even after 10% oversampling a few fraud rows could otherwise
     dominate the batch loss.

### Config knobs (`config.yaml` → `dataset:`)

| Key | Meaning |
| --- | --- |
| `csv_path` | Explicit path to `creditcard.csv` (skips auto-resolution) |
| `cache_dir` | Where the CSV + parsed `.npz` cache live (default `.cache/`) |
| `test_size` | Stratified train/test split fraction (default `0.2`) |
| `oversample_fraud_ratio` | Target train-split fraud ratio; `null`/`0` trains on the raw ~1:578 imbalance |
| `max_train_samples` / `max_test_samples` | Cap dataset size for fast local iteration; set both to `null` to train on the full 284,807 rows (slower ledger init on first run, most realistic) |

## What "a sample" is here

There are no images — **each sample is one transaction (a row)**, and the
model input **is** the 1-D feature vector. WeightsLab carries that vector
through gRPC as a `raw_data` stat of type `vector`, so `inputs`, `labels`/
`target`, and `metadata` all reach the UI. The 30 raw features are also
exposed as **sortable columns** via the dataset's `get_items()` metadata
contract (`preload_metadata=True`), so the List Exploration view shows real
tabular columns (`Time`, `V1`, `Amount`, …) alongside the tracked stats below.
Everything you do on MNIST — sort, lock, histograms, discard/restore, neuron
ops — works the same way, because those operate on the per-sample ledger, not
on pixels.

## What you'll see in the UI

| Signal / column | Meaning |
| --- | --- |
| feature columns (`Time`, `V1`..`V28`, `Amount`) | The 30 raw transaction features, sortable |
| `train-loss-CE`, `test-loss-CE` | Weighted cross-entropy per split |
| `train-metric-Precision/Recall/F1`, `test-metric-Precision/Recall/F1` | Fraud-class precision/recall/F1 per split — `train-*` is the current training batch (logged every step), `test-*` is the full eval pass |
| `train-metric-AveragePrecision`, `test-metric-AveragePrecision` | Area under the precision-recall curve — the metric to watch given the imbalance |
| `train_metric/Accuracy_per_sample`, `test_metric/Accuracy_per_sample` | Per-transaction correctness (0/1) — included for reference, but nearly meaningless alone: predicting "always legit" already scores ~99.8% at this prevalence |
| `train_metric/Fraud_caught_per_sample`, `test_metric/Fraud_caught_per_sample` | 1 when a true fraud was correctly flagged |
| `train_metric/False_alarm_per_sample`, `test_metric/False_alarm_per_sample` | 1 when a legit transaction was incorrectly flagged as fraud |
| `target`, `prediction` columns | Per-sample truth/pred to sort/lock in List view |

## Test it

```bash
# Fast, offline unit tests (pure PyTorch/pandas, no gRPC server, no Kaggle
# download — a small synthetic CSV with the real schema stands in):
python -m pytest test_fraud_detection.py -v

# End-to-end integration check (needs weightslab installed): drives the
# tracked loaders + watched loss/metrics + gRPC server against the same
# synthetic-CSV fixture, then asserts the ledger dataframe the UI reads has
# per-sample rows, every feature as a column, target/prediction/loss, and a
# live gRPC endpoint.
python verify_integration.py
```

Neither test requires Kaggle credentials or the real download — they exercise
the exact same loading/caching/splitting/oversampling code path against a
small in-memory CSV with the real dataset's column schema. Only `main.py`
itself needs the real `creditcard.csv`.
