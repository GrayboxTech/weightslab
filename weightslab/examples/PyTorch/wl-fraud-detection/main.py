"""WeightsLab example: bank card-transaction fraud detection (tabular, PyTorch).

A small MLP binary classifier over synthetic transaction features, wired into
WeightsLab so the run streams per-sample stats (loss, prediction, target,
discard state) to the UI. Because the data is tabular, this is a natural fit for
the List Exploration (tabular) view — sort by loss or prediction to triage the
transactions the model finds hardest.

Run:
    cd weightslab/examples/PyTorch/wl-fraud-detection
    python main.py

The dataset/model live in ``utils/`` (pure PyTorch) so they can be unit tested
without the gRPC backend — see ``test_fraud_detection.py``.
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

from torchmetrics.classification import Accuracy

import weightslab as wl
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)

from utils.data import FraudDataset, NUM_FEATURES
from utils.model import FraudMLP


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Train / Test steps
# -----------------------------------------------------------------------------
def train(loader, model, optimizer, criterion_mlt, device):
    """Single training step using the tracked dataloader + watched loss."""
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

    return total_loss.detach().cpu().item()


def test(loader, model, criterion_mlt, metric_mlt, device, test_loader_len):
    """Full evaluation pass over the test loader, logging per-sample signals."""
    losses = torch.tensor(0.0, device=device)

    for (inputs, ids, labels) in loader:
        with guard_testing_context:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1, keepdim=True)

            loss_batch = criterion_mlt(outputs, labels, batch_ids=ids, preds=preds)
            losses += torch.mean(loss_batch)
            metric_mlt.update(outputs, labels)

            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            acc_per_sample = (preds_flat == labels_flat).float()
            fraud_caught_per_sample = ((preds_flat == 1) & (labels_flat == 1)).float()

            signals = {
                "test_metric/Accuracy_per_sample": acc_per_sample,
                "test_metric/Fraud_caught_per_sample": fraud_caught_per_sample,
            }
            wl.save_signals(
                preds_raw=outputs,
                targets=labels,
                batch_ids=ids,
                signals=signals,
                preds=preds,
            )

    loss = losses / max(1, test_loader_len)
    metric = metric_mlt.compute() * 100

    return loss.detach().cpu().item(), metric.detach().cpu().item()


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

    # Data (synthetic tabular fraud stream) — no download needed.
    dataset_cfg = parameters.get("dataset", {})
    seed = int(dataset_cfg.get("seed", 0))
    n_train = int(dataset_cfg.get("n_train", 4000))
    n_test = int(dataset_cfg.get("n_test", 1000))

    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    _train_dataset = FraudDataset(n_train, seed=seed, max_samples=train_cfg.get("max_samples"))
    _test_dataset = FraudDataset(n_test, seed=seed + 1, max_samples=test_cfg.get("max_samples"))

    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", 32),
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
        batch_size=test_cfg.get("batch_size", 128),
        shuffle=test_cfg.get("shuffle", False),
        is_training=False,
        compute_hash=False,
        preload_labels=True,
        preload_metadata=True,
        enable_h5_persistence=enable_h5_persistence,
    )

    # Losses & metrics (watched objects – they log themselves).
    # Class weighting counters the ~12% fraud prevalence so the minority class
    # actually drives the gradient.
    class_weights = torch.tensor(
        parameters.get("class_weights", [1.0, 4.0]), dtype=torch.float32, device=device
    )
    train_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(weight=class_weights, reduction="none"),
        flag="loss", signal_name="train-loss-CE", log=True)
    test_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(weight=class_weights, reduction="none"),
        flag="loss", signal_name="test-loss-CE", log=True)

    metric = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=2).to(device),
        flag="metric", signal_name="metric-ACC", log=True)

    # Start WeightsLab services (gRPC only, no CLI).
    wl.serve(serving_grpc=parameters.get("serving_grpc", False))

    print("=" * 60)
    print(" STARTING FRAUD-DETECTION TRAINING")
    print(f" Evaluation every {eval_full_to_train_steps_ratio} steps")
    print(f" Dataset splits: train={len(_train_dataset)}, test={len(_test_dataset)}")
    print(f" Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")

    if tqdm_display:
        train_range = tqdm.tqdm(
            range(training_steps_to_do) if training_steps_to_do is not None else itertools.count(),
            desc="Training",
            bar_format="{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}] {bar} | {postfix}",
            ncols=140,
            position=0,
            leave=True,
        )
    else:
        train_range = range(training_steps_to_do) if training_steps_to_do is not None else itertools.count()

    # ================
    # Training Loop
    wl.start_training(timeout=3)

    train_loss = None
    test_loss, test_metric = None, None
    test_loader_len = len(test_loader)
    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step

        train_loss = train(train_loader, model, optimizer, train_criterion, device)

        if age > 0 and age % eval_full_to_train_steps_ratio == 0:
            test_loss, test_metric = test(
                test_loader, model, test_criterion, metric, device, test_loader_len
            )

        if age > 0 and age % write_export_ratio == 0:
            wl.write_history()
            wl.write_dataframe()

        if verbose and not tqdm_display:
            import sys
            msg = f"Step {train_step} (Age {age}): Loss={train_loss:.4f}"
            if test_loss is not None:
                msg += f" | Test={test_loss:.4f} ({test_metric:.1f}%)"
            sys.stdout.write(f"\r{msg:<100}")
            sys.stdout.flush()
        elif tqdm_display:
            postfix_parts = [f"train_loss={train_loss:.4f}"]
            if test_loss is not None:
                postfix_parts.append(f"test_loss={test_loss:.4f}")
            if test_metric is not None:
                postfix_parts.append(f"test_acc={test_metric:.1f}%")
            train_range.set_postfix_str(" | ".join(postfix_parts))

    print("\n" + "=" * 60)
    print(f" Training completed in {time.time() - start_time:.2f} seconds")
    print(f" Logs saved to: {log_dir}")
    print("=" * 60)

    wl.write_history()
    wl.write_dataframe()
    wl.keep_serving()
