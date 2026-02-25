import os
import time
import tempfile

import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl

import weightslab as wl

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchmetrics.classification import Accuracy

from weightslab.baseline_models.pytorch.models import FashionCNN as CNN
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context
)


# =============================================================================
# Custom MNIST Dataset with Filepath Metadata
# =============================================================================
class MNISTCustomDataset(Dataset):
    """
    Custom MNIST dataset that includes filepath metadata for each image.
    
    Returns tuples of (image, label, filepath) where filepath is stored
    as metadata that can be tracked by WeightsLab.
    """
    
    def __init__(self, root, train=True, download=False, transform=None):
        """
        Args:
            root (str): Root directory where MNIST data is stored
            train (bool): If True, use training data; else use test data
            download (bool): If True, download the data if not present
            transform (callable, optional): Optional transform to be applied on images
        """
        # Load the standard MNIST dataset
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=None  # We'll apply transform manually to track filepath
        )
        self.transform = transform
        self.train = train
        self.root = root
        
        # Build filepath mapping for each sample
        self._build_filepath_mapping()
    
    def _build_filepath_mapping(self):
        """Build a mapping of sample index to filepath."""
        self.filepaths = {}
        
        # For each index, construct a meaningful filepath
        # MNIST doesn't have original individual files, so we create virtual paths
        for idx in range(len(self.mnist)):
            label = self.mnist.targets[idx].item() if hasattr(self.mnist.targets[idx], 'item') else self.mnist.targets[idx]
            split = 'train' if self.train else 'test'
            
            # Create a virtual filepath that identifies the image
            virtual_path = os.path.join(
                'MNIST',
                'processed',
                split,
                f'class_{label}',
                f'sample_{idx:05d}.pt'
            )
            self.filepaths[idx] = virtual_path
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, idx, label)
        """
        image, label = self.mnist[idx]
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        return image, idx, label
    
    
class LitMNIST(pl.LightningModule):
    def __init__(self, model, optim, train_criterion_wl=None, val_criterion_wl=None, metric_wl=None):
        super().__init__()

        # Model hyperparameters
        self.model = model

        # WeightsLab tracked loss and metrics
        self.train_criterion_wl = train_criterion_wl
        self.val_criterion_wl = val_criterion_wl
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
            if self.train_criterion_wl is not None:
                loss_batch = self.train_criterion_wl(
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
            if self.val_criterion_wl is not None:
                self.val_criterion_wl(
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

    # Data (MNIST train/val/test)
    # Use data_root from config if provided, otherwise fall back to log_dir/data
    if parameters.get("data_root"):
        should_download = False
        if not os.path.exists(parameters["data_root"]):
            print(f"Warning: data_root {parameters['data_root']} does not exist. Will attempt to download to this location.")
            should_download = True
        data_root = parameters["data_root"]
    else:
        data_root = os.path.join(parameters["root_log_dir"], "data")
        should_download = True
        print(f"Downloading data to {data_root}")
    os.makedirs(data_root, exist_ok=True)

    _train_dataset = MNISTCustomDataset(
        root=data_root,
        train=True,
        download=should_download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    _val_dataset = MNISTCustomDataset(
        root=data_root,
        train=False,
        download=should_download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
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
    train_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="train-loss-CE", log=True)
    val_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="val-loss-CE", log=True)
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
    L_model = LitMNIST(model=model_wl, optim=optimizer, train_criterion_wl=train_criterion, val_criterion_wl=val_criterion, metric_wl=metric)
    
    # Start WeightsLab services
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", False),
        serving_cli=parameters.get("serving_cli", False),
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
