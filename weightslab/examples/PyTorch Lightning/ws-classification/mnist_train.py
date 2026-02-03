import os
import time
import tempfile

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

import weightslab as wl
from weightslab.baseline_models.pytorch.models import FashionCNN as CNN
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context
)


class LitMNIST(pl.LightningModule):
    def __init__(self, model, optim, criterion_wl=None, metric_wl=None):
        super().__init__()

        # Model hyperparameters
        self.model = model

        # WeightsLab tracked loss and metrics
        self.criterion_wl = criterion_wl
        self.metric_wl = metric_wl

        # Training hyperparameters
        self.optimizer = optim

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        with guard_training_context:
            x, ids, y = batch
            logits = self(x)  # forward pass
            preds = torch.argmax(logits, dim=1)
            
            # WeightsLab tracked loss
            if self.criterion_wl is not None:
                loss_batch = self.criterion_wl(
                    logits.float(),
                    y.long(),
                    batch_ids=ids,
                    preds=preds
                )
                loss = loss_batch.mean()

            return loss

    def validation_step(self, batch):
        with guard_testing_context:
            x, ids, y = batch
            logits = self(x)
            preds = torch.argmax(logits, dim=1)
            
            # WeightsLab tracked loss - auto logs wi. WL SDK
            if self.criterion_wl is not None:
                self.criterion_wl(
                    logits.float(),
                    y.long(),
                    batch_ids=ids,
                    preds=preds
                )
            
            # Update WeightsLab metric
            if self.metric_wl is not None:
                self.metric_wl.update(logits, y)
            
            # Per-sample accuracy for WeightsLab
            acc_per_sample = (preds == y).float()
            acc_reversed_per_sample = (preds != y).float()
            
            # Log per-sample metrics to WeightsLab
            signals = {
                "val_metric/Accuracy_per_sample": acc_per_sample,
                "val_metric/Inverse_Accuracy_per_sample": acc_reversed_per_sample,
            }
            wl.save_signals(
                preds_raw=logits,
                targets=y,
                batch_ids=ids,
                signals=signals,
                preds=preds,
            )

    def configure_optimizers(self):
        return self.optimizer


# Main training function
def main():
    start_time = time.time()

    # Load hyperparameters (from YAML if present)
    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    parameters = parameters or {}

    # ---- sensible defaults / normalization ----
    parameters.setdefault("experiment_name", "mnist_lightning")
    parameters.setdefault("device", "auto")
    parameters.setdefault("max_epochs", 5)

    # Device selection
    if parameters.get("device", "auto") == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        parameters["device"] = device_str
    device = parameters["device"]

    # Logging dir
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = tmp_dir
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    log_dir = parameters["root_log_dir"]
    max_epochs = parameters.get("max_epochs", 5)
    enable_h5_persistence = parameters.get("enable_h5_persistence", True)

    # Hyperparameters (must use 'hyperparameters' flag for trainer services / UI)
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        defaults=parameters,
        poll_interval=1.0,
    )

    # Data directory handling
    if parameters.get("data_dir"):
        user_path = parameters["data_dir"]
        files_here = os.path.exists(os.path.join(user_path, "train-images-idx3-ubyte")) or \
                     os.path.exists(os.path.join(user_path, "train-images-idx3-ubyte.gz"))

        if files_here:
            data_root = os.path.dirname(os.path.dirname(user_path))
            print(f"Using existing data from {user_path}")
        else:
            check_child = os.path.join(user_path, "MNIST", "raw")
            files_child = os.path.exists(os.path.join(check_child, "train-images-idx3-ubyte")) or \
                          os.path.exists(os.path.join(check_child, "train-images-idx3-ubyte.gz"))
            if files_child:
                data_root = user_path
                print(f"Using existing data found in {check_child}")
            else:
                data_root = user_path
                print(f"Data not found, will download to {data_root}")
    else:
        data_root = os.path.join(parameters["root_log_dir"], "data")
        print(f"Downloading data to {data_root}")

    os.makedirs(data_root, exist_ok=True)

    # Create datasets
    _train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    _val_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Read data config
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    val_cfg = parameters.get("data", {}).get("val_loader", {})

    train_bs = train_cfg.get("batch_size", 64)
    val_bs = val_cfg.get("batch_size", 64)

    train_shuffle = train_cfg.get("shuffle", True)
    val_shuffle = val_cfg.get("shuffle", False)

    # Create WeightsLab tracked loaders
    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=train_bs,
        shuffle=train_shuffle,
        is_training=True,
        compute_hash=False,
        preload_labels=False,
        enable_h5_persistence=enable_h5_persistence
    )
    val_loader = wl.watch_or_edit(
        _val_dataset,
        flag="data",
        loader_name="val_loader",
        batch_size=val_bs,
        shuffle=val_shuffle,
        is_training=False,
        compute_hash=False,
        preload_labels=False,
        enable_h5_persistence=enable_h5_persistence
    )

    # WeightsLab tracked loss and metrics
    criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="loss-CE", log=True)

    metric = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=10).to(device),
        flag="metric", signal_name="metric-ACC", log=True
    )

    # Create model - wrap CNN with WeightsLab, then pass to Lightning module
    _model = CNN().to(device)
    model_wl = wl.watch_or_edit(_model, flag="model", device=device)
    _optimizer = torch.optim.Adam(model_wl.parameters(), lr=parameters.get("optimizer", {}).get("lr", 0.001))
    optimizer = wl.watch_or_edit(
        _optimizer,
        flag="optimizer"
    )

    # Generate the lightning module
    L_model = LitMNIST(model=model_wl, optim=optimizer, criterion_wl=criterion, metric_wl=metric)
    
    # Start WeightsLab services
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", False),
        serving_cli=parameters.get("serving_cli", False),
        serving_ui=parameters.get("serving_ui", False),
    )

    print("=" * 60)
    print("🚀 STARTING TRAINING (PyTorch Lightning)")
    print(f"📊 Max epochs: {max_epochs}")
    print(f"📦 Dataset splits: train={len(_train_dataset)}, val={len(_val_dataset)}")
    print(f"💾 Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")

    # PyTorch Lightning Trainer
    pl.seed_everything(42, workers=True)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device if device in ["cpu", "cuda", "mps"] else "auto",
        devices=1,

        # Log part are disabled as WL SDK handles logging
        log_every_n_steps=0,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(L_model, train_loader, val_loader)

    print("\n" + "=" * 60)
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep the main thread alive to allow background serving threads to run
    wl.keep_serving()


if __name__ == "__main__":
    main()
