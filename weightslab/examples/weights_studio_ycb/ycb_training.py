import os
import time
import tempfile
import logging

import yaml
import tqdm
import torch
import itertools
import torch.nn as nn
import torch.optim as optim

import weightslab as wl
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torchmetrics.classification import Accuracy

from weightslab.utils.board import Dash as Logger
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context
)

# Setup logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)


# Simple CNN for YCB (RGB 3xHxW, variable num_classes)
class YCBCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 28):
        super().__init__()
        # Exposed for WeightsLab's ModelInterface
        # Batch dimension will be added by ModelInterface if needed; they call
        # torch.randn(model.input_shape), so we give it (1, C, H, W)
        self.input_shape = (1, 3, image_size, image_size)

        # Input: (B, 3, image_size, image_size)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # After two 2x2 pools, spatial size is image_size / 4
        feat_h = image_size // 4
        feat_w = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feat_h * feat_w, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self._age = 0  # simple age counter for logging

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        self._age += 1
        return out

    def get_age(self):
        return self._age


# Train / Test functions
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


def test(loader, model, criterion_mlt, metric_mlt, device, loader_len):
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

    loss = losses / loader_len
    metric = metric_mlt.compute() * 100

    return loss.detach().cpu().item(), metric.detach().cpu().item()


if __name__ == "__main__":
    # --- 1) Load hyperparameters from YAML (if present) ---
    config_path = os.path.join(os.path.dirname(__file__), "ycb_training_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    # Defaults
    parameters.setdefault("experiment_name", "ycb_cnn")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 1000)
    parameters.setdefault("eval_full_to_train_steps_ratio", 50)
    parameters["is_training"] = True  # start in running mode
    parameters.setdefault("number_of_workers", 4)

    exp_name = parameters["experiment_name"]

    # --- 2) Device selection ---
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = parameters["device"]

    # --- 3) Logging directory ---
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = os.path.join(tmp_dir, "logs")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    tqdm_display = parameters.get('tqdm_display', True)
    tqdm_display_eval = parameters.get('tqdm_display_eval', True)
    verbose = parameters.get('verbose', True)
    log_dir = parameters["root_log_dir"]
    max_steps = parameters["training_steps_to_do"]
    eval_every = parameters["eval_full_to_train_steps_ratio"]

    # --- 4) Register logger + hyperparameters ---
    logger = Logger()
    wl.watch_or_edit(logger, flag="logger", name=exp_name, log_dir=log_dir)

    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )

    # --- 5) Data (YCB train/val using ImageFolder, RGB) ---
    data_root = parameters.get("data", {}).get(
        "data_dir",
        os.path.join(parameters["root_log_dir"], "data", "ycb_datasets"),
    )
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    image_size = parameters.get("image_size", 128)
    common_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    # Load subsample of datasets for quick testing
    _train_dataset = datasets.ImageFolder(root=train_dir, transform=common_transform)
    _train_dataset = Subset(_train_dataset, list(range(min(1000, len(_train_dataset)))))
    
    _test_dataset = datasets.ImageFolder(root=val_dir, transform=common_transform)
    _test_dataset = Subset(_test_dataset, list(range(min(200, len(_test_dataset)))))

    num_classes = len(datasets.ImageFolder(root=train_dir, transform=common_transform).classes)

    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        name="train_loader",
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=train_cfg.get("train_shuffle", True),
        is_training=True,
        compute_hash=True
    )
    test_loader = wl.watch_or_edit(
        _test_dataset,
        flag="data",
        name="test_loader",
        batch_size=test_cfg.get("batch_size", 16),
        shuffle=test_cfg.get("test_shuffle", False),
        compute_hash=True
    )

    # --- 6) Model, optimizer, losses, metric ---
    _model = YCBCNN(num_classes=num_classes, image_size=image_size)
    model = wl.watch_or_edit(_model, flag="model", name=exp_name, device=device)

    lr = parameters.get("optimizer", {}).get("lr", 0.01)
    _optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = wl.watch_or_edit(_optimizer, flag="optimizer", name=exp_name)

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
        Accuracy(task="multiclass", num_classes=num_classes).to(device),
        flag="metric",
        name="test_metric/mlt_metric",
        log=True,
    )

    # --- 7) Start WeightsLab services (UI + gRPC) ---
    wl.serve(
        serving_ui=True,
        root_directory=log_dir,
        
        serving_cli=True,

        serving_grpc=True,
        n_workers_grpc=parameters.get("number_of_workers"),
    )

    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print(f"📈 Total steps: {max_steps}")
    print(f"🔄 Evaluation every {eval_every} steps")
    print(f"💾 Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")

    # --- 8) Resume training automatically ---
    # pause_controller.resume()

    # ================
    # 9. Training Loop
    print("\nStarting Training...")
    max_steps = parameters.get('training_steps_to_do', 6666)
    train_range = tqdm.tqdm(itertools.count(), desc="Training") if tqdm_display else itertools.count()
    test_loader_len = len(test_loader)  # Store length before wrapping with tqdm
    test_loader = tqdm.tqdm(test_loader, desc="Evaluating") if tqdm_display_eval else test_loader
    test_loss, test_metric = None, None
    for train_step in train_range:
        train_loss = train(train_loader, model, optimizer, train_criterion_mlt, device)

        test_loss, test_metric = None, None
        if train_step % eval_every == 0:
            test_loss, test_metric = test(
                test_loader,
                model,
                test_criterion_mlt,
                test_metric_mlt,
                device,
                test_loader_len
            )

        if train_step % 10 == 0 or test_loss is not None:
            status = f"[Step {train_step:5d}/{max_steps}] Train Loss: {train_loss:.4f}"
            if test_loss is not None:
                status += (
                    f" | Test Loss: {test_loss:.4f} | Test Acc: {test_metric:.2f}%"
                )
            print(status)

    print("\n" + "=" * 60)
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep server alive for UI
    print("\nServer is still running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
