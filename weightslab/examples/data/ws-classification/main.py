import itertools
import os
import time
import tempfile
import logging

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import weightslab as wl

from torchvision import datasets, transforms
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms

from weightslab.baseline_models.pytorch.models import FashionCNN as CNN
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context
)


# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Train / Test functions
# -----------------------------------------------------------------------------
def train(loader, model, optimizer, criterion_mlt, device):
    """Single training step using the tracked dataloader + watched loss."""
    with guard_training_context:
        (inputs, ids, labels) = next(loader)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Infer
        optimizer.zero_grad()
        preds_raw = model(inputs)

        # Preds
        if preds_raw.ndim == 1:
            preds = (preds_raw > 0.0).long()
        else:
            preds = preds_raw.argmax(dim=1, keepdim=True)

        # Loss is a watched object => pass metadata for logging/stats
        loss_batch_mlt = criterion_mlt(
            preds_raw.float(),
            labels.long(),
            batch_ids=ids,
            preds=preds
        )
        total_loss = loss_batch_mlt.mean()  # Final scalar loss

        # Model
        total_loss.backward()
        optimizer.step()

    return total_loss.detach().cpu().item()


def test(loader, model, criterion_mlt, metric_mlt, device, test_loader_len):
    """Full evaluation pass over the test loader."""
    losses = 0.0

    for (inputs, ids, labels) in loader:
        with guard_testing_context, torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Infer
            outputs = model(inputs)

            # Preds
            if outputs.ndim == 1:
                preds = (outputs > 0.0).long()
            else:
                preds = outputs.argmax(dim=1, keepdim=True)

            # Compute signals
            loss_batch = criterion_mlt(
                outputs,
                labels,
                batch_ids=ids,
                preds=preds,
            )
            losses += torch.mean(loss_batch)
            metric_mlt.update(outputs, labels)

            # Per-sample accuracy: 1.0 if correct, else 0.0
            preds_flat = preds.view(-1)
            acc_per_sample = (preds_flat == labels.view(-1)).float()

            # Log per-sample metric alongside signals; persists via the storer
            wl.save_signals(
                preds_raw=outputs,
                targets=labels,
                batch_ids=ids,
                signals={"valOrtest/accuracy_per_sample": acc_per_sample},
                preds=preds,
            )


    loss = losses / test_loader_len
    metric = metric_mlt.compute() * 100

    return loss.detach().cpu().item(), metric.detach().cpu().item()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    # Load hyperparameters (from YAML if present)
    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    parameters = parameters or {}

    # ---- sensible defaults / normalization ----
    parameters.setdefault("experiment_name", "mnist_cnn")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 1000)
    parameters.setdefault("eval_full_to_train_steps_ratio", 50)

    # Experiment name
    exp_name = parameters["experiment_name"]

    # Device selection
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]

    # Logging dir
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = os.path.join(tmp_dir, "logs")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    verbose = parameters.get('verbose', True)
    log_dir = parameters["root_log_dir"]
    tqdm_display = parameters.get("tqdm_display", True)
    eval_every = parameters.get("eval_full_to_train_steps_ratio", 50)
    enable_h5_persistence = parameters.get("enable_h5_persistence", True)

    # Hyperparameters (must use 'hyperparameters' flag for trainer services / UI)
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )

    # Model
    model = CNN().to(device)

    # Optimizer
    lr = parameters.get("optimizer", {}).get("lr", 0.01)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Data (MNIST train/val/test)
    _full_train_dataset = datasets.MNIST(
        root=os.path.join(parameters["root_log_dir"], "data"),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    _test_dataset = datasets.MNIST(
        root=os.path.join(parameters["root_log_dir"], "data"),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    # Split train into train + val (80/20 split)
    val_split = parameters.get("data", {}).get("val_split", 0.2)
    train_size = int((1.0 - val_split) * len(_full_train_dataset))
    val_size = len(_full_train_dataset) - train_size

    _train_dataset, _val_dataset = torch.utils.data.random_split(
        _full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Read data config for all loaders
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    val_cfg = parameters.get("data", {}).get("val_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    train_bs = train_cfg.get("batch_size", 16)
    val_bs = val_cfg.get("batch_size", 16)
    test_bs = test_cfg.get("batch_size", 16)

    train_shuffle = train_cfg.get("shuffle", True)
    val_shuffle = val_cfg.get("shuffle", False)
    test_shuffle = test_cfg.get("shuffle", False)

    # Create tracked loaders for train, val, and test
    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        name="train_loader",
        batch_size=train_bs,
        shuffle=train_shuffle,
        is_training=True,
        compute_hash=False,
        enable_h5_persistence=enable_h5_persistence
    )
    val_loader = wl.watch_or_edit(
        _val_dataset,
        flag="data",
        name="val_loader",
        batch_size=val_bs,
        shuffle=val_shuffle,
        is_training=False,
        compute_hash=False,
        enable_h5_persistence=enable_h5_persistence
    )
    test_loader = wl.watch_or_edit(
        _test_dataset,
        flag="data",
        name="test_loader",
        batch_size=test_bs,
        shuffle=test_shuffle,
        is_training=False,
        compute_hash=False,
        enable_h5_persistence=enable_h5_persistence
    )

    # Losses & metrics (watched objects – they log themselves)
    train_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss",
        name="train_mlt_loss/CE",
        log=True,
    )
    val_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss",
        name="val_mlt_loss/CE",
        log=True,
    )
    test_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss",
        name="test_mlt_loss/CE",
        log=True,
    )

    val_metric_mlt = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=10).to(device),
        flag="metric",
        name="val_metric/Accuracy",
        log=True,
    )
    test_metric_mlt = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=10).to(device),
        flag="metric",
        name="test_metric/Accuracy",
        log=True,
    )

    # Start WeightsLab services (gRPC only, no CLI)
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", False),
        serving_cli=parameters.get("serving_cli", False),
    )

    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print(f"🔄 Evaluation every {eval_every} steps")
    print(f"� Dataset splits: train={len(_train_dataset)}, val={len(_val_dataset)}, test={len(_test_dataset)}")
    print(f"💾 Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")

    train_range = tqdm.tqdm(itertools.count(), desc="Training") if tqdm_display else itertools.count()
    val_loader_len = len(val_loader)  # Store length before wrapping with tqdm
    test_loader_len = len(test_loader)

    train_loss = None
    val_loss, val_metric = None, None
    test_loss, test_metric = None, None
    for train_step in train_range:
        # Train one step
        train_loss = train(train_loader, model, optimizer, train_criterion_mlt, device)

        # Periodic validation and test evaluation
        if train_step > 0 and train_step % eval_every == 0:
            # Validate
            val_loader_iter = tqdm.tqdm(val_loader, desc="Validating") if tqdm_display else val_loader
            val_loss, val_metric = test(
                val_loader_iter,
                model,
                val_criterion_mlt,
                val_metric_mlt,
                device,
                val_loader_len
            )

            # Test (less frequent or same as val)
            test_loader_iter = tqdm.tqdm(test_loader, desc="Testing") if tqdm_display else test_loader
            test_loss, test_metric = test(
                test_loader_iter,
                model,
                test_criterion_mlt,
                test_metric_mlt,
                device,
                test_loader_len
            )

        # Verbose
        if verbose and not tqdm_display:
            print(
                f"Training.. " +
                f"Step {train_step}: " +
                f"| Train Loss: {train_loss:.4f} " +
                (f"| Val Loss: {val_loss:.4f} " if val_loss is not None else '') +
                (f"| Val Acc: {val_metric:.2f}% " if val_metric is not None else '') +
                (f"| Test Loss: {test_loss:.4f} " if test_loss is not None else '') +
                (f"| Test Acc: {test_metric:.2f}% " if test_metric is not None else '')
            )
        elif tqdm_display:
            train_range.set_description(f"Step")
            train_range.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}" if val_loss is not None else "N/A",
                val_acc=f"{val_metric:.2f}%" if val_metric is not None else "N/A",
                test_loss=f"{test_loss:.4f}" if test_loss is not None else "N/A",
                test_acc=f"{test_metric:.2f}%" if test_metric is not None else "N/A"
            )

    print("\n" + "=" * 60)
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep the main thread alive to allow background serving threads to run
    wl.keep_serving()