"""WeightsLab example: real bank card-transaction fraud detection (tabular).

An MLP binary classifier over the real Kaggle "Credit Card Fraud Detection"
(ULB) dataset — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud —
wired into WeightsLab so the run streams per-sample stats (loss, prediction,
target, discard state) to the UI. Being tabular, it's the natural companion to
the List Exploration (tabular) view — sort by loss or prediction to triage the
transactions the model finds hardest.

Getting the data (one-time, see utils/kaggle_data.py for details):
    pip install kagglehub
    python -c "import kagglehub; kagglehub.login()"
  ...or manually download creditcard.csv from the Kaggle page above and set
  `dataset.csv_path` in config.yaml (or env WL_FRAUD_CSV_PATH).

Run:
    cd weightslab/examples/PyTorch/wl-fraud-detection
    python main.py

The dataset/model live in ``utils/`` (pure PyTorch/pandas) so the data
pipeline can be unit tested without downloading the real CSV or starting the
gRPC backend — see ``test_fraud_detection.py``.
"""

import itertools
import os
import time
import logging
import tempfile

import yaml
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics.classification import Precision, Recall, F1Score, AveragePrecision

import weightslab as wl
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)

from utils.data import load_creditcard_fraud, compute_class_weights, NUM_FEATURES
from utils.model import FraudMLP


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Train / Test steps
# -----------------------------------------------------------------------------
def train(loader, model, optimizer, criterion_mlt, metrics, device):
    """Single training step: watched loss plus per-batch metrics/signals.

    Mirrors ``test()`` so the train split gets the same precision/recall/F1/AP
    curves and per-sample signals as eval — logged under ``train-metric-*`` and
    ``train_metric/*`` — instead of only the loss.
    """
    with guard_training_context:
        (inputs, ids, labels) = next(loader)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds_raw = model(inputs)
        preds = preds_raw.argmax(dim=1, keepdim=True)

        loss_batch_mlt = criterion_mlt(
            preds_raw.float(),
            labels.long(),
            batch_ids=ids,
            preds=preds,
        )
        total_loss = loss_batch_mlt.mean()

        total_loss.backward()
        optimizer.step()

        # Per-batch training metrics + per-sample signals. Detach the probs: the
        # step is done, so metric bookkeeping must not hang onto the graph.
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        fraud_probs = torch.softmax(preds_raw, dim=1)[:, 1].detach()
        metrics["precision"].update(preds_flat, labels_flat)
        metrics["recall"].update(preds_flat, labels_flat)
        metrics["f1"].update(preds_flat, labels_flat)
        metrics["ap"].update(fraud_probs, labels_flat)

        acc_per_sample = (preds_flat == labels_flat).float()
        fraud_caught_per_sample = ((preds_flat == 1) & (labels_flat == 1)).float()
        false_alarm_per_sample = ((preds_flat == 1) & (labels_flat == 0)).float()

        signals = {
            "train_metric/Accuracy_per_sample": acc_per_sample,
            "train_metric/Fraud_caught_per_sample": fraud_caught_per_sample,
            "train_metric/False_alarm_per_sample": false_alarm_per_sample,
        }
        wl.save_signals(
            preds_raw=preds_raw,
            targets=labels,
            batch_ids=ids,
            signals=signals,
            preds=preds,
        )

    results = {name: m.compute().item() * 100 for name, m in metrics.items()}
    for m in metrics.values():
        m.reset()  # per-step train batch metric, not a running total

    return total_loss.detach().cpu().item(), results


def test(loader, model, criterion_mlt, metrics, device, test_loader_len):
    """Full evaluation pass over the (naturally-imbalanced) test loader.

    Reports precision/recall/F1/average-precision for the fraud class instead
    of plain accuracy — at ~0.17% fraud prevalence, "always predict legit"
    already scores >99.8% accuracy, so accuracy alone would hide whether the
    model does anything useful at all.
    """
    losses = torch.tensor(0.0, device=device)

    for (inputs, ids, labels) in loader:
        with guard_testing_context:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1, keepdim=True)
            fraud_probs = torch.softmax(outputs, dim=1)[:, 1]

            loss_batch = criterion_mlt(outputs, labels, batch_ids=ids, preds=preds)
            losses += torch.mean(loss_batch)

            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            metrics["precision"].update(preds_flat, labels_flat)
            metrics["recall"].update(preds_flat, labels_flat)
            metrics["f1"].update(preds_flat, labels_flat)
            metrics["ap"].update(fraud_probs, labels_flat)

            acc_per_sample = (preds_flat == labels_flat).float()
            fraud_caught_per_sample = ((preds_flat == 1) & (labels_flat == 1)).float()
            false_alarm_per_sample = ((preds_flat == 1) & (labels_flat == 0)).float()

            signals = {
                "test_metric/Accuracy_per_sample": acc_per_sample,
                "test_metric/Fraud_caught_per_sample": fraud_caught_per_sample,
                "test_metric/False_alarm_per_sample": false_alarm_per_sample,
            }
            wl.save_signals(
                preds_raw=outputs,
                targets=labels,
                batch_ids=ids,
                signals=signals,
                preds=preds,
            )

    loss = losses / max(1, test_loader_len)
    results = {name: m.compute().item() * 100 for name, m in metrics.items()}
    for m in metrics.values():
        m.reset()  # each eval pass reports on that pass alone, not a running total

    return loss.detach().cpu().item(), results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    # Load hyperparameters (from YAML if present).
    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    parameters = parameters or {}

    # ---- sensible defaults / normalization ----
    parameters.setdefault("experiment_name", "fraud_detection_mlp")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 1000000)
    parameters.setdefault("eval_full_to_train_steps_ratio", 100)

    exp_name = parameters["experiment_name"]

    # Hyperparameters (must use 'hyperparameters' flag for trainer services / UI).
    wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

    # Device selection
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]

    # Logging dir
    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = tempfile.mkdtemp()
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    verbose = parameters.get("verbose", True)
    log_dir = parameters["root_log_dir"]
    tqdm_display = parameters.get("tqdm_display", True)
    eval_full_to_train_steps_ratio = parameters.get("eval_full_to_train_steps_ratio", 100)
    write_export_ratio = parameters.get("write_export_ratio", 100)
    enable_h5_persistence = parameters.get("enable_h5_persistence", True)
    training_steps_to_do = parameters.get("training_steps_to_do", 1000)

    # Model
    _model = FraudMLP(in_features=NUM_FEATURES, num_classes=2).to(device)
    model = wl.watch_or_edit(_model, flag="model", device=device)

    # Optimizer
    lr = parameters.get("optimizer", {}).get("lr", 0.005)
    _optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = wl.watch_or_edit(_optimizer, flag="optimizer")

    # Data — the real Kaggle CSV (downloaded/cached on first run, see
    # utils/kaggle_data.py). This is the slow, one-time step; every run after
    # that hits the cached, pre-parsed .npz instead of re-parsing the CSV.
    dataset_cfg = parameters.get("dataset", {})
    print("Loading creditcard.csv (first run downloads + caches it; this may take a while)...")
    _train_dataset, _test_dataset = load_creditcard_fraud(
        csv_path=dataset_cfg.get("csv_path"),
        cache_dir=dataset_cfg.get("cache_dir"),
        seed=int(dataset_cfg.get("seed", 0)),
        test_size=float(dataset_cfg.get("test_size", 0.2)),
        oversample_fraud_ratio=dataset_cfg.get("oversample_fraud_ratio", 0.1),
        max_train_samples=dataset_cfg.get("max_train_samples"),
        max_test_samples=dataset_cfg.get("max_test_samples"),
    )

    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", 256),
        shuffle=train_cfg.get("shuffle", True),
        is_training=True,
        compute_hash=False,
        preload_labels=True,
        preload_metadata=True,
        enable_h5_persistence=enable_h5_persistence,
    )
    test_loader = wl.watch_or_edit(
        _test_dataset,
        flag="data",
        loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 512),
        shuffle=test_cfg.get("shuffle", False),
        is_training=False,
        compute_hash=False,
        preload_labels=True,
        preload_metadata=True,
        enable_h5_persistence=enable_h5_persistence,
    )

    # Losses & metrics (watched objects – they log themselves).
    # Class weights are computed from the actual training-split labels (post
    # oversampling), not hardcoded, since the real prevalence (~0.17%) is far
    # more extreme than a toy dataset and swings with the oversample_fraud_ratio
    # and max_train_samples knobs above.
    class_weights_cfg = parameters.get("class_weights")
    if class_weights_cfg:
        class_weights = torch.tensor(class_weights_cfg, dtype=torch.float32, device=device)
    else:
        train_labels = _train_dataset.labels.numpy()
        auto_weights = compute_class_weights(train_labels, cap=parameters.get("class_weight_cap", 20.0))
        class_weights = torch.tensor(auto_weights, dtype=torch.float32, device=device)
    print(f"Class weights [legit, fraud] = {class_weights.tolist()}")

    train_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(weight=class_weights, reduction="none"),
        flag="loss", signal_name="train-loss-CE", log=True)
    test_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(weight=class_weights, reduction="none"),
        flag="loss", signal_name="test-loss-CE", log=True)

    # One watched Precision/Recall/F1/AP set per split, logged under
    # <split>-metric-* so the train and eval curves show up side by side.
    def _build_metrics(split):
        return {
            "precision": wl.watch_or_edit(
                Precision(task="binary").to(device), flag="metric",
                signal_name=f"{split}-metric-Precision", log=True),
            "recall": wl.watch_or_edit(
                Recall(task="binary").to(device), flag="metric",
                signal_name=f"{split}-metric-Recall", log=True),
            "f1": wl.watch_or_edit(
                F1Score(task="binary").to(device), flag="metric",
                signal_name=f"{split}-metric-F1", log=True),
            "ap": wl.watch_or_edit(
                AveragePrecision(task="binary").to(device), flag="metric",
                signal_name=f"{split}-metric-AveragePrecision", log=True),
        }

    train_metrics = _build_metrics("train")
    metrics = _build_metrics("test")

    # Start WeightsLab services (gRPC only, no CLI).
    wl.serve(serving_grpc=parameters.get("serving_grpc", False))

    print("=" * 60)
    print(" STARTING FRAUD-DETECTION TRAINING (real Kaggle creditcard.csv)")
    print(f" Evaluation every {eval_full_to_train_steps_ratio} steps")
    print(f" Dataset splits: train={len(_train_dataset)}, test={len(_test_dataset)}")
    print(f" Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")

    if tqdm_display:
        train_range = tqdm.tqdm(
            range(training_steps_to_do) if training_steps_to_do != None else itertools.count(),
            desc="Training",
            bar_format="{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}] {bar} | {postfix}",
            ncols=140,
            position=0,
            leave=True,
        )
    else:
        train_range = range(training_steps_to_do) if training_steps_to_do != None else itertools.count()

    # ================
    # Training Loop
    wl.start_training(timeout=3)

    train_loss = None
    test_loss, test_metrics = None, None
    test_loader_len = len(test_loader)
    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step

        train_loss, train_metrics_vals = train(
            train_loader, model, optimizer, train_criterion, train_metrics, device
        )

        if age > 0 and age % eval_full_to_train_steps_ratio == 0:
            test_loss, test_metrics = test(
                test_loader, model, test_criterion, metrics, device, test_loader_len
            )

        if age > 0 and age % write_export_ratio == 0:
            wl.write_history()
            wl.write_dataframe()

        if verbose and not tqdm_display:
            import sys
            msg = (f"Step {train_step} (Age {age}): Loss={train_loss:.4f} "
                   f"(F1={train_metrics_vals['f1']:.1f}% AP={train_metrics_vals['ap']:.1f}%)")
            if test_loss is not None:
                msg += (f" | Test={test_loss:.4f} "
                        f"(P={test_metrics['precision']:.1f}% R={test_metrics['recall']:.1f}% "
                        f"F1={test_metrics['f1']:.1f}% AP={test_metrics['ap']:.1f}%)")
            sys.stdout.write(f"\r{msg:<160}")
            sys.stdout.flush()
        elif tqdm_display:
            postfix_parts = [
                f"train_loss={train_loss:.4f}",
                f"tr_F1={train_metrics_vals['f1']:.1f}%",
                f"tr_AP={train_metrics_vals['ap']:.1f}%",
            ]
            if test_loss is not None:
                postfix_parts.append(f"test_loss={test_loss:.4f}")
                postfix_parts.append(f"te_P={test_metrics['precision']:.1f}%")
                postfix_parts.append(f"te_R={test_metrics['recall']:.1f}%")
                postfix_parts.append(f"te_F1={test_metrics['f1']:.1f}%")
                postfix_parts.append(f"te_AP={test_metrics['ap']:.1f}%")
            train_range.set_postfix_str(" | ".join(postfix_parts))

    print("\n" + "=" * 60)
    print(f" Training completed in {time.time() - start_time:.2f} seconds")
    print(f" Logs saved to: {log_dir}")
    print("=" * 60)

    wl.write_history()
    wl.write_dataframe()
    wl.keep_serving()
