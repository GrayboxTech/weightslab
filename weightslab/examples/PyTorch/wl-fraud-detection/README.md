# WeightsLab — Bank Fraud Detection (tabular, pure PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/wl-fraud-detection.ipynb)

A small, fully-runnable **tabular binary-classification** example: an MLP learns
to flag fraudulent bank card transactions, streaming per-sample loss /
prediction / accuracy to the WeightsLab UI. Being tabular, it's the natural
companion to the **List Exploration (tabular) view** — sort by `loss` or
`prediction` to triage the transactions the model finds hardest.

Everything is plain PyTorch + NumPy — no external download. The dataset is
generated in-process so runs are reproducible and offline.

## Quick start

```bash
cd weightslab/examples/PyTorch/wl-fraud-detection
pip install -r requirements.txt
python main.py
```

Then open the UI (e.g. `http://localhost:5173`), switch the Data Exploration
board to **List** view, press play, and sort columns to explore.

## The data

`make_synthetic_fraud(n, seed)` (in `utils/data.py`) generates a reproducible
stream of transactions. Each row has **16 numeric features** (`amount`,
`old_balance`, `merchant_risk`, `geo_distance_km`, `is_foreign`,
`num_prior_disputes`, …) and a binary label (`0` legit, `1` fraud, ~12%
prevalence). Fraud rows come from shifted distributions (larger amounts,
off-hours activity, higher merchant risk, device changes, larger geo jumps).
Features are standardized and reshaped to a `1x4x4` heatmap so the grid renders
a thumbnail; the model flattens it back to 16.

### Using a real dataset

Swap `FraudDataset` for a loader over a real CSV to go from demo to reality:

* **Kaggle Credit Card Fraud** (ULB) — `creditcard.csv`, 284k transactions with
  anonymized `V1..V28` features + `Amount` + `Class`.
  <https://www.kaggle.com/mlg-ulb/creditcardfraud>
* **PaySim** mobile-money fraud simulator.
  <https://www.kaggle.com/ealaxi/paysim1>

Keep the `__getitem__` contract `(features_as_image, idx, label)` and adjust
`NUM_FEATURES` / the reshape.

## What "a sample" is here

There are no images — **each sample is one transaction (a row)**, and the model
input **is** the 1-D feature vector (not a reshaped image). WeightsLab carries
that vector through gRPC as a `raw_data` stat of type `vector` (the actual
values), so `inputs`, `labels`/`target` and `metadata` all reach the UI. The 16
raw features are also exposed as **sortable columns** via the dataset's
`get_items()` metadata contract (`preload_metadata=True`), so the List
Exploration view shows real tabular columns (`amount`, `merchant_risk`,
`geo_distance_km`, …) alongside the tracked stats below. Everything you do on
MNIST — sort, lock, histograms, discard/restore, neuron ops — works the same
way, because those operate on the per-sample ledger, not on pixels.

## What you'll see in the UI

| Signal / column                          | Meaning                                        |
| ---------------------------------------- | ---------------------------------------------- |
| feature columns (`amount`, `merchant_risk`, …) | The 16 raw transaction features, sortable |
| `train-loss-CE`, `test-loss-CE`          | Weighted cross-entropy per split               |
| `metric-ACC`                             | Overall accuracy                               |
| `test_metric/Accuracy_per_sample`        | Per-transaction correctness (0/1)              |
| `test_metric/Fraud_caught_per_sample`    | 1 when a true fraud was correctly flagged      |
| `target`, `prediction` columns           | Per-sample truth/pred to sort/lock in List view |

Class weights (`[1.0, 4.0]`, config) up-weight the fraud class against the ~12%
prevalence so the minority class drives the gradient.

## Test it

```bash
# Fast, offline unit tests (pure PyTorch, no gRPC server):
python -m pytest test_fraud_detection.py -v

# End-to-end integration check (needs weightslab installed): drives the tracked
# loaders + watched loss + gRPC server, then asserts the ledger dataframe the UI
# reads has per-sample rows, every feature as a column, target/prediction/loss,
# and a live gRPC endpoint.
python verify_integration.py
```

The unit tests cover the dataset contract, reproducibility, class balance, the
model forward pass, `get_items` metadata columns, and that a few optimizer steps
reduce the loss (final accuracy > 90%). `verify_integration.py` proves WeightsLab
is fully wired in tabular mode.
