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

from weightslab.baseline_models.pytorch.models import FashionCNN as CNN
from weightslab.utils.board import Dash as Logger
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
    pause_controller,
)


# Setup logging
logging.basicConfig(level=logging.ERROR)


# -----------------------------------------------------------------------------
# Train / Test functions
# -----------------------------------------------------------------------------
def train(loader, model, optimizer, criterion_mlt, device="cpu"):
    """Single training step using the tracked dataloader + watched loss."""
    with guard_training_context:
        (inputs, ids, labels) = next(loader)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if outputs.ndim == 1:
            preds = (outputs > 0.0).long()
        else:
            preds = outputs.argmax(dim=1, keepdim=True)

        # Loss is a watched object => pass metadata for logging/stats
        loss_batch = criterion_mlt(
            outputs.float(),
            labels.long(),
            model_age=model.get_age(),
            batch_ids=ids,
            preds=preds,
        )
        loss = loss_batch.mean()

        loss.backward()
        optimizer.step()

    return loss.detach().cpu().item()


def test(loader, model, criterion_mlt, metric_mlt, device):
    """Full evaluation pass over the test loader."""
    losses = 0.0

    with guard_testing_context, torch.no_grad():
        for (inputs, ids, labels) in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if outputs.ndim == 1:
                preds = (outputs > 0.0).long()
            else:
                preds = outputs.argmax(dim=1, keepdim=True)

            loss_batch = criterion_mlt(
                outputs,
                labels,
                model_age=model.get_age(),
                batch_ids=ids,
                preds=preds,
            )
            losses += torch.mean(loss_batch)
            metric_mlt.update(outputs, labels)

    loss = losses / len(loader)
    metric = metric_mlt.compute() * 100

    return loss.detach().cpu().item(), metric.detach().cpu().item()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    # 1) Start WeightsLab services (gRPC only, no CLI)
    wl.serve(
        serving_grpc=True,
        n_workers_grpc=6,
        port_grpc=50051,
        serving_cli=False,     # no CLI TCP server, no extra terminal
        host_cli="127.0.0.1",
        port_cli=0,
        launch_cli=False,
    )

    # 2) Load hyperparameters (from YAML if present)
    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), "mnist_training_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}

    parameters = parameters or {}
    # ---- sensible defaults / normalization ----
    parameters.setdefault("experiment_name", "mnist_cnn")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 1000)
    parameters.setdefault("eval_full_to_train_steps_ratio", 50)
    # FORCE training to start in "running" mode
    parameters["is_training"] = True

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

    log_dir = parameters["root_log_dir"]
    tqdm_display = parameters.get("tqdm_display", True)
    max_steps = parameters.get("training_steps_to_do", 1000)
    eval_every = parameters.get("eval_full_to_train_steps_ratio", 50)

    # 3) Register components in the GLOBAL_LEDGER via watch_or_edit
    # Logger
    logger = Logger()
    wl.watch_or_edit(logger, flag="logger", name=exp_name, log_dir=log_dir)

    # Hyperparameters (must use 'hyperparameters' flag for trainer services / UI)
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )

    # Model
    _model = CNN()
    model = wl.watch_or_edit(_model, flag="model", name=exp_name, device=device)

    # Optimizer
    lr = parameters.get("optimizer", {}).get("lr", 0.01)
    _optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = wl.watch_or_edit(_optimizer, flag="optimizer", name=exp_name)

    # Data (MNIST train/test)
    _train_dataset = datasets.MNIST(
        root=os.path.join(parameters["root_log_dir"], "data"),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
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
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    # Read data config in unified style: data.train_loader / data.test_loader
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    train_bs = train_cfg.get("batch_size", 16)
    test_bs = test_cfg.get("batch_size", 16)
    train_shuffle = train_cfg.get("train_shuffle", True)
    test_shuffle = test_cfg.get("test_shuffle", False)

    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        name="train_loader",
        batch_size=train_bs,
        shuffle=train_shuffle,
        is_training=True,
    )
    test_loader = wl.watch_or_edit(
        _test_dataset,
        flag="data",
        name="test_loader",
        batch_size=test_bs,
        shuffle=test_shuffle,
    )

    # Losses & metric (watched objects – they log themselves)
    train_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss",
        name="train_loss/mlt_loss",
        log=True,
    )
    test_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss",
        name="test_loss/mlt_loss",
        log=True,
    )
    test_metric_mlt = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=10).to(device),
        flag="metric",
        name="test_metric/mlt_metric",
        log=True,
    )

    # 4) HARD-UNPAUSE global training state so guard_training_context can't block
    try:
        pause_controller.resume()
        # In case resume() has a bug / race, force the internals too
        if hasattr(pause_controller, "_paused"):
            pause_controller._paused = False
        if hasattr(pause_controller, "_event"):
            pause_controller._event.set()
        print(">>> Forced training to RUN state (pause_controller).")
    except Exception as e:
        print(f">>> Failed to force unpause: {e}")

    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print(f"📈 Total steps: {max_steps}")
    print(f"🔄 Evaluation every {eval_every} steps")
    print(f"💾 Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")

    train_range = tqdm.trange(max_steps, dynamic_ncols=True) if tqdm_display else range(max_steps)

    for train_step in train_range:
        # Train one step
        train_loss = train(train_loader, model, optimizer, train_criterion_mlt, device)

        # Periodic full eval
        test_loss, test_metric = None, None
        if train_step % eval_every == 0:
            test_loss, test_metric = test(
                test_loader,
                model,
                test_criterion_mlt,
                test_metric_mlt,
                device,
            )

        # Verbose logging
        if train_step % 10 == 0 or test_loss is not None:
            status = f"[Step {train_step:5d}/{max_steps}] Train Loss: {train_loss:.4f}"
            if test_loss is not None:
                status += f" | Test Loss: {test_loss:.4f} | Test Acc: {test_metric:.2f}%"
            print(status)

    print("\n" + "=" * 60)
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep the script running to serve requests for the UI
    print("\nServer is still running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
