# WeightsLab Examples

Runnable, self-contained examples wired into WeightsLab with the same
`wl.watch_or_edit(...)` / `wl.serve(...)` pattern. Each **card** links to the
example folder (`main.py` + `config.yaml`) and, where available, a one-click
**Google Colab** notebook.

> Open the UI in a separate terminal to watch any of these live:
> ```bash
> weightslab ui launch     # http://localhost:5173
> ```

## Tabular

The model input is a **feature vector** (no images): a sample is a **row**, and
every feature is a **sortable column** in the List Exploration view. WeightsLab
sends the input vector to the UI through gRPC as a `vector` — see
[the tabular notes](#how-tabular-works) below.

| Example | Task | Run it | Colab |
| --- | --- | --- | --- |
| [**Bank Fraud Detection**](PyTorch/wl-fraud-detection) | Binary classification on 16 synthetic transaction features (~12% fraud). MLP. | `cd PyTorch/wl-fraud-detection && python main.py` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/wl-fraud-detection.ipynb) |
| [**Advertising CTR Recommendation**](PyTorch/wl-ads-recommendation) | Click-through-rate prediction (8 categorical + 8 numeric fields, ~20% CTR). Wide & Deep. | `cd PyTorch/wl-ads-recommendation && python main.py` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/wl-ads-recommendation.ipynb) |

Both datasets are generated in-process (no download). Real drop-in replacements
are documented in each folder's README — Kaggle Credit Card Fraud / PaySim
(fraud); Criteo / Avazu / MovieLens (ads).

## Computer vision (PyTorch)

| Example | Task | Run it | Colab |
| --- | --- | --- | --- |
| [Image Classification](PyTorch/wl-classification) | MNIST digit classification. | `weightslab start example --cls` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/wl-classification.ipynb) |
| [Clustering](PyTorch/wl-clustering) | Feature/embedding clustering. | `weightslab start example --clus` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/wl-clustering.ipynb) |
| [Object Detection](PyTorch/wl-detection) | Single-shot detector on Penn-Fudan. | `weightslab start example --det` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/wl-detection.ipynb) |
| [Segmentation](PyTorch/wl-segmentation) | Semantic segmentation (BDD subset). | `weightslab start example --seg` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/wl-segmentation.ipynb) |
| [Generation](PyTorch/wl-generation) | Generative model example. | `weightslab start example --gen` | — |

## How tabular works

For tabular examples the dataset:

1. Returns the **feature vector** from `__getitem__` as the model input — no
   fake image.
2. Implements `get_items(idx, include_metadata, include_labels, include_images)`
   returning `(input, uid, target, metadata)`; the `metadata` dict becomes one
   **sortable column per feature** (loaded via `preload_metadata=True`).

WeightsLab transmits the input vector to the UI through gRPC as a `raw_data`
stat of `type="vector"` carrying the exact values, alongside `target`,
`prediction`, and the per-feature columns. Sorting, locking, histograms,
discard/restore and neuron ops all work exactly as on image datasets — they
operate on the per-sample ledger, not on pixels.

Each folder also ships a `verify_integration.py` that drives the full stack
(tracked loaders + watched loss + gRPC) and asserts the feature values reach the
UI over gRPC.
