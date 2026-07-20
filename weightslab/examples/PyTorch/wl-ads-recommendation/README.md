# WeightsLab — Advertising CTR Recommendation (tabular, pure PyTorch)

A fully-runnable **click-through-rate (CTR) prediction** example — the core of
an advertising recommendation system. A **Wide & Deep** model learns to predict
`P(click)` for `(user, ad, context)` impressions, streaming per-impression loss /
prediction / accuracy to the WeightsLab UI. Being tabular, it's a natural fit
for the **List Exploration (tabular) view** — sort by `loss` or `prediction` to
inspect the impressions the model ranks best and worst.

Everything is plain PyTorch + NumPy — no external download. The impressions are
generated in-process so runs are reproducible and offline.

## Quick start

```bash
cd weightslab/examples/PyTorch/wl-ads-recommendation
pip install -r requirements.txt
python main.py
```

Then open the UI (e.g. `http://localhost:5173`), switch the Data Exploration
board to **List** view, press play, and sort columns to explore.

## The data & model

`make_synthetic_ctr(n, seed)` (in `utils/data.py`) generates reproducible ad
impressions with:

* **8 categorical fields** — `user_segment`, `ad_category`, `device_type`, `os`,
  `publisher`, `placement`, `region`, `hour_bucket` (embedded by the model).
* **8 numeric features** — `ad_position`, `bid_price`, `user_age`,
  `session_depth`, `historical_ctr`, … (standardized).
* a binary `clicked` label at a calibrated **~20% CTR**, driven by per-field
  effects plus an `ad_category × user_segment` interaction.

The 16 field values are packed into a `1x4x4` tensor per impression (heatmap
thumbnail in the grid); the model unpacks them back into indices + numerics.

**Model — Wide & Deep** (`utils/model.py`, after Cheng et al., 2016):
* *Deep*: per-field embeddings ⊕ numeric → MLP (generalizes feature interactions).
* *Wide*: first-order per-category effects + linear numeric term (memorizes
  strong direct signals).
The two logit heads are summed. Related architectures: DeepFM, Factorization
Machines.

### Using a real dataset

Swap `AdsCTRDataset` for a loader over a real CTR log and set
`CATEGORICAL_CARDINALITIES` to your vocab sizes:

* **Criteo Display Advertising** — 13 numeric + 26 categorical fields.
  <https://www.kaggle.com/c/criteo-display-ad-challenge>
* **Avazu CTR** — <https://www.kaggle.com/c/avazu-ctr-prediction>
* **MovieLens** (recommender variant) — <https://grouplens.org/datasets/movielens/>

Keep the `__getitem__` contract `(packed_features_as_image, idx, label)`.

## What you'll see in the UI

| Signal / column                          | Meaning                                       |
| ---------------------------------------- | --------------------------------------------- |
| `train-loss-CE`, `test-loss-CE`          | Weighted cross-entropy per split              |
| `metric-ACC`                             | Overall accuracy                              |
| `test_metric/PredictedCTR_per_sample`    | Model's predicted `P(click)` per impression   |
| `test_metric/Accuracy_per_sample`        | Per-impression correctness (0/1)              |
| `loss`, `prediction`, `target` columns   | Per-sample columns to sort/lock in List view  |

## Test it

```bash
python -m pytest test_ads_recommendation.py -v
# or:  python test_ads_recommendation.py
```

Pure-PyTorch smoke tests (no gRPC server): schema, reproducibility, calibrated
CTR, pack/unpack roundtrip, model forward pass, and that training reduces loss
and **ranks real clicks above non-clicks** on a held-out split (rank-AUC > 0.6).
